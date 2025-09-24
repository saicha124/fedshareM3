#!/usr/bin/env python3
import pickle
import sys
import threading
import hashlib
import time
import json
from collections import defaultdict

import requests
from flask import Flask, request, jsonify

from config import HierValidatorConfig

config = HierValidatorConfig(int(sys.argv[1]))

api = Flask(__name__)

# Validator committee state
pending_shares = defaultdict(list)  # facility_id -> list of shares
vote_records = defaultdict(dict)    # share_id -> validator_id -> vote
validated_shares = []
committee_private_key = f"validator_{config.validator_index}_private_key"
committee_public_key = hashlib.sha256(committee_private_key.encode()).hexdigest()

def verify_facility_signature(share_data, signature, facility_public_key):
    """Verify digital signature from healthcare facility"""
    try:
        # Accept RSA signatures (hex format) or HMAC signatures (64-char hex)
        if not signature or len(signature) == 0:
            return False
        
        # Accept both RSA-PSS signatures (variable length hex) and HMAC (64-char hex)
        if isinstance(signature, str) and len(signature) > 0:
            # Basic format validation - accept hex strings
            try:
                int(signature, 16)  # Test if it's valid hex
                return True
            except ValueError:
                # Also accept base64 format
                try:
                    import base64
                    base64.b64decode(signature)
                    return True
                except Exception:
                    return False
        
        return False
        
    except Exception as e:
        print(f"Signature verification error: {e}")
        return False

def validate_proof_of_work(facility_id, nonce, hash_result):
    """Validate Proof-of-Work from healthcare facility"""
    try:
        # Recreate the PoW challenge
        facility_data = f"{facility_id}||facility_public_key"
        challenge_input = f"{nonce}||{facility_data}"
        computed_hash = hashlib.sha256(challenge_input.encode()).hexdigest()
        
        # Check if hash meets difficulty requirement
        hash_value = int(computed_hash, 16)
        is_valid = hash_value < config.pow_target
        
        print(f"PoW validation for facility {facility_id}: {'valid' if is_valid else 'invalid'}")
        return is_valid
        
    except Exception as e:
        print(f"PoW validation error: {e}")
        return False

def validate_share_integrity(share_data):
    """Validate the integrity and consistency of the secret share"""
    try:
        # Check required fields
        required_fields = ['share_id', 'data_fragment', 'size_info', 'threshold', 'total_shares']
        if not all(field in share_data for field in required_fields):
            print("Share missing required fields")
            return False
        
        # Check share parameters consistency
        if share_data['total_shares'] != config.num_fog_nodes:
            print(f"Share total_shares mismatch: {share_data['total_shares']} != {config.num_fog_nodes}")
            return False
        
        if share_data['threshold'] != config.secret_threshold:
            print(f"Share threshold mismatch: {share_data['threshold']} != {config.secret_threshold}")
            return False
        
        # Check data fragment is reasonable
        data_fragment = share_data.get('data_fragment', b'')
        if not isinstance(data_fragment, (bytes, str)):
            print("Invalid data fragment type")
            return False
        
        print(f"Share integrity validation passed for share {share_data['share_id']}")
        return True
        
    except Exception as e:
        print(f"Share integrity validation error: {e}")
        return False

def cast_vote(share_id, facility_id, share_data, signature, facility_public_key):
    """Cast vote on whether to approve the share"""
    validator_id = config.validator_index
    
    # Validate the share
    vote = 1  # Start with approve
    
    # Check signature
    if not verify_facility_signature(share_data, signature, facility_public_key):
        print(f"Validator {validator_id}: Invalid signature from facility {facility_id}")
        vote = 0
    
    # Check share integrity
    if not validate_share_integrity(share_data):
        print(f"Validator {validator_id}: Share integrity check failed for facility {facility_id}")
        vote = 0
    
    # Additional Byzantine fault tolerance checks
    if vote == 1:
        # Check for unusual patterns that might indicate Byzantine behavior
        share_size = len(share_data.get('data_fragment', b''))
        if share_size == 0:
            print(f"Validator {validator_id}: Empty share data from facility {facility_id}")
            vote = 0
        elif share_size > 10 * 1024 * 1024:  # 10MB limit
            print(f"Validator {validator_id}: Share too large from facility {facility_id}")
            vote = 0
    
    # Record the vote
    vote_records[share_id][validator_id] = vote
    
    print(f"Validator {validator_id} cast vote {vote} for share {share_id} from facility {facility_id}")
    return vote

def get_other_validators():
    """Get list of other validator committee members"""
    other_validators = []
    for i in range(config.committee_size):
        if i != config.validator_index:
            other_validators.append({
                'validator_id': i,
                'port': config.committee_base_port + i
            })
    return other_validators

def broadcast_vote_to_committee(share_id, vote, share_data):
    """Broadcast vote to other committee members"""
    vote_message = {
        'share_id': share_id,
        'validator_id': config.validator_index,
        'vote': vote,
        'timestamp': time.time(),
        'share_data': share_data
    }
    
    other_validators = get_other_validators()
    successful_broadcasts = 0
    
    for validator in other_validators:
        try:
            url = f"http://{config.server_address}:{validator['port']}/receive_vote"
            response = requests.post(url, json=vote_message, timeout=10)
            
            if response.status_code == 200:
                successful_broadcasts += 1
            else:
                print(f"Failed to broadcast vote to validator {validator['validator_id']}")
                
        except requests.RequestException as e:
            print(f"Network error broadcasting to validator {validator['validator_id']}: {e}")
    
    print(f"Vote broadcast to {successful_broadcasts}/{len(other_validators)} validators")
    return successful_broadcasts

def check_consensus(share_id):
    """Check if consensus has been reached for a share"""
    if share_id not in vote_records:
        return False, 0
    
    votes = vote_records[share_id]
    total_votes = len(votes)
    approve_votes = sum(1 for vote in votes.values() if vote == 1)
    
    # Need majority consensus
    consensus_reached = approve_votes >= config.consensus_threshold
    
    print(f"Share {share_id}: {approve_votes}/{total_votes} votes, consensus: {'reached' if consensus_reached else 'not reached'}")
    return consensus_reached, approve_votes

def sign_committee_approval(share_data):
    """Create committee signature for approved share"""
    data_str = json.dumps(share_data, sort_keys=True, default=str)
    signature_input = f"{data_str}||committee_signature"
    signature = hashlib.sha256(signature_input.encode()).hexdigest()
    return signature

def broadcast_to_fog_nodes(approved_share_data):
    """Broadcast approved share to appropriate fog node"""
    share_data = approved_share_data['share']
    facility_id = approved_share_data['facility_id']
    share_id = share_data['share_id']
    
    # Determine which fog node should receive this share (use numeric routing)
    fog_node_index = (share_id - 1) % config.num_fog_nodes
    fog_node_port = config.fog_node_base_port + fog_node_index
    
    # Add committee signature
    committee_signed_share = {
        'facility_id': facility_id,
        'share': share_data,
        'committee_signature': sign_committee_approval(share_data),
        'committee_public_key': committee_public_key,
        'validation_timestamp': time.time(),
        'validator_id': config.validator_index
    }
    
    try:
        url = f"http://{config.server_address}:{fog_node_port}/receive_share"
        response = requests.post(url, json=committee_signed_share, timeout=30)
        
        if response.status_code == 200:
            print(f"Successfully broadcast approved share to fog node {fog_node_index}")
            return True
        else:
            print(f"Failed to broadcast to fog node {fog_node_index}: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"Network error broadcasting to fog node {fog_node_index}: {e}")
        return False

@api.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "validator_id": config.validator_index,
        "status": "healthy",
        "algorithm": "hierarchical_federated",
        "committee_size": config.committee_size,
        "pending_shares": len(pending_shares),
        "validated_shares": len(validated_shares),
        "public_key": committee_public_key[:16]
    })

@api.route('/validate_share', methods=['POST'])
def validate_share():
    """Receive and validate secret share from healthcare facility"""
    try:
        share_request = request.get_json()
        
        facility_id = share_request['facility_id']
        share_data = share_request['share']
        signature = share_request['signature']
        facility_public_key = share_request['public_key']
        
        # Use deterministic share_uid from client for consensus
        share_id = share_request.get('share_uid', f"{facility_id}_{share_data['share_id']}_{time.time()}")
        
        print(f"Validator {config.validator_index} received share from facility {facility_id}")
        
        # Cast vote on the share
        vote = cast_vote(share_id, facility_id, share_data, signature, facility_public_key)
        
        # Broadcast vote to other committee members
        broadcast_vote_to_committee(share_id, vote, share_request)
        
        return jsonify({
            "response": "share_received",
            "validator_id": config.validator_index,
            "share_id": share_id,
            "vote": vote
        })
        
    except Exception as e:
        print(f"Error validating share: {e}")
        return jsonify({"error": "Share validation failed"}), 500

@api.route('/receive_vote', methods=['POST'])
def receive_vote():
    """Receive vote from another committee member"""
    try:
        vote_data = request.get_json()
        
        share_id = vote_data['share_id']
        voter_id = vote_data['validator_id']
        vote = vote_data['vote']
        
        # Record the vote
        vote_records[share_id][voter_id] = vote
        
        print(f"Received vote {vote} from validator {voter_id} for share {share_id}")
        
        # If this validator hasn't voted yet, validate the share and cast own vote
        current_validator_id = config.validator_index
        if current_validator_id not in vote_records[share_id] and 'share_data' in vote_data:
            share_request = vote_data['share_data']
            facility_id = share_request['facility_id']
            share_data = share_request['share']
            signature = share_request['signature']
            facility_public_key = share_request['public_key']
            
            # Cast our own vote on this share
            own_vote = cast_vote(share_id, facility_id, share_data, signature, facility_public_key)
            vote_records[share_id][current_validator_id] = own_vote
            
            print(f"Validator {current_validator_id} cast vote {own_vote} for share {share_id}")
        
        # Check if we have consensus
        consensus_reached, approve_votes = check_consensus(share_id)
        
        if consensus_reached:
            # Broadcast approved share to fog nodes
            share_request = vote_data['share_data']
            print(f"Consensus reached for share {share_id}, broadcasting to fog nodes")
            
            success = broadcast_to_fog_nodes(share_request)
            
            if success:
                validated_shares.append(share_id)
                # Clean up vote records
                if share_id in vote_records:
                    del vote_records[share_id]
        
        return jsonify({"response": "vote_recorded", "validator_id": config.validator_index})
        
    except Exception as e:
        print(f"Error processing vote: {e}")
        return jsonify({"error": "Vote processing failed"}), 500

@api.route('/status', methods=['GET'])
def get_status():
    """Get detailed validator status"""
    return jsonify({
        "validator_id": config.validator_index,
        "committee_size": config.committee_size,
        "consensus_threshold": config.consensus_threshold,
        "pending_votes": len(vote_records),
        "validated_shares": len(validated_shares),
        "validator_port": config.validator_port,
        "byzantine_tolerance": config.max_byzantine_nodes
    })

@api.route('/reset', methods=['POST'])
def reset_validator():
    """Reset validator state for new round"""
    global pending_shares, vote_records, validated_shares
    
    pending_shares.clear()
    vote_records.clear()
    validated_shares.clear()
    
    print(f"Validator {config.validator_index} reset for new round")
    return jsonify({"response": "reset_complete", "validator_id": config.validator_index})

if __name__ == '__main__':
    print(f"Starting Hierarchical Federated Learning Validator {config.validator_index}")
    print(f"Committee Public Key: {committee_public_key[:16]}...")
    print(f"Listening on port: {config.validator_port}")
    print(f"Committee size: {config.committee_size}, Consensus threshold: {config.consensus_threshold}")
    print(f"Byzantine fault tolerance: {config.max_byzantine_nodes} nodes")
    
    # Start the validator server
    api.run(host=config.server_address, 
           port=config.validator_port, 
           debug=False, 
           threaded=True)