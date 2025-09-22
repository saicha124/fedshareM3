#!/usr/bin/env python3
import pickle
import threading
import time
import hashlib
import json
import secrets

import requests
from flask import Flask, request, jsonify

from config import HierTrustedAuthorityConfig

config = HierTrustedAuthorityConfig()

api = Flask(__name__)

# Trusted Authority state
registered_facilities = {}
attribute_universe = ['hospital', 'clinic', 'research_center', 'north', 'south', 'east', 'west']
master_secret_key = secrets.token_hex(32)
public_key = hashlib.sha256(master_secret_key.encode()).hexdigest()
issued_keys = {}

def setup_cp_abe():
    """Initialize CP-ABE system with public/master secret keys"""
    print("Initializing CP-ABE system...")
    print(f"Public Key: {public_key[:16]}...")
    print(f"Attribute Universe: {attribute_universe}")
    return public_key, master_secret_key

def verify_proof_of_work(facility_id, nonce, hash_result, public_key_ref):
    """Verify Proof-of-Work from facility registration"""
    try:
        # Recreate the challenge
        facility_data = f"{facility_id}||{public_key_ref}"
        challenge_input = f"{nonce}||{facility_data}"
        computed_hash = hashlib.sha256(challenge_input.encode()).hexdigest()
        
        # Verify hash meets difficulty requirement
        hash_value = int(computed_hash, 16)
        is_valid = hash_value < config.pow_target and hash_result == computed_hash
        
        print(f"PoW verification for facility {facility_id}: {'valid' if is_valid else 'invalid'}")
        return is_valid
        
    except Exception as e:
        print(f"PoW verification error: {e}")
        return False

def generate_attribute_key(facility_id, attributes):
    """Generate CP-ABE attribute-based secret key for facility"""
    # Simplified CP-ABE key generation
    # In production, use proper CP-ABE library
    
    attribute_string = json.dumps(sorted(attributes.items()))
    key_input = f"{facility_id}||{attribute_string}||{master_secret_key}"
    
    secret_key = {
        'facility_id': facility_id,
        'attributes': attributes,
        'key_data': hashlib.sha256(key_input.encode()).hexdigest(),
        'issued_timestamp': time.time(),
        'issuer': 'trusted_authority'
    }
    
    return secret_key

def encrypt_model_with_cp_abe(model_data, access_policy):
    """Encrypt global model using CP-ABE with access policy"""
    # Simplified CP-ABE encryption
    # In production, use proper CP-ABE library
    
    policy_string = json.dumps(access_policy, sort_keys=True)
    encryption_key = hashlib.sha256(f"{policy_string}||{master_secret_key}".encode()).hexdigest()
    
    encrypted_model = {
        'ciphertext': hashlib.sha256(model_data).hexdigest(),  # Simulated encryption
        'access_policy': access_policy,
        'encrypted_data': model_data,  # In production, this would be properly encrypted
        'encryption_timestamp': time.time(),
        'public_key': public_key
    }
    
    return encrypted_model

def check_facility_attributes(facility_attributes, access_policy):
    """Check if facility attributes satisfy access policy"""
    # Simplified policy evaluation
    # In production, use proper policy evaluation engine
    
    required_type = access_policy.get('facility_type')
    required_region = access_policy.get('region')
    required_certified = access_policy.get('certified')
    
    if required_type and facility_attributes.get('facility_type') != required_type:
        return False
    
    if required_region and facility_attributes.get('region') != required_region:
        return False
    
    if required_certified is not None and facility_attributes.get('certified') != required_certified:
        return False
    
    return True

def distribute_to_facility(facility_id, encrypted_model):
    """Distribute encrypted model to specific healthcare facility"""
    try:
        facility_port = config.facility_base_port + facility_id
        url = f"http://{config.client_address}:{facility_port}/receive_global_model"
        
        response = requests.post(url, json=encrypted_model, timeout=30)
        
        if response.status_code == 200:
            print(f"Successfully distributed model to facility {facility_id}")
            return True
        else:
            print(f"Failed to distribute to facility {facility_id}: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"Network error distributing to facility {facility_id}: {e}")
        return False

@api.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "server_id": "trusted_authority",
        "status": "healthy",
        "algorithm": "hierarchical_federated",
        "registered_facilities": len(registered_facilities),
        "issued_keys": len(issued_keys),
        "public_key": public_key[:16]
    })

@api.route('/register_facility', methods=['POST'])
def register_facility():
    """Register healthcare facility with PoW verification"""
    try:
        registration_data = request.get_json()
        
        facility_id = registration_data['facility_id']
        facility_public_key = registration_data['public_key']
        nonce = registration_data['nonce']
        hash_result = registration_data['hash_result']
        attributes = registration_data['attributes']
        
        print(f"Processing registration for facility {facility_id}")
        
        # Verify Proof-of-Work
        if not verify_proof_of_work(facility_id, nonce, hash_result, facility_public_key):
            print(f"PoW verification failed for facility {facility_id}")
            return jsonify({"error": "Proof-of-Work verification failed"}), 401
        
        # Validate attributes
        valid_attributes = all(attr in attribute_universe for attr in attributes.values() if isinstance(attr, str))
        if not valid_attributes:
            print(f"Invalid attributes for facility {facility_id}")
            return jsonify({"error": "Invalid facility attributes"}), 400
        
        # Generate CP-ABE secret key
        secret_key = generate_attribute_key(facility_id, attributes)
        
        # Store registration
        registered_facilities[facility_id] = {
            'facility_id': facility_id,
            'public_key': facility_public_key,
            'attributes': attributes,
            'registration_timestamp': time.time(),
            'status': 'registered'
        }
        
        issued_keys[facility_id] = secret_key
        
        print(f"Facility {facility_id} registered successfully")
        print(f"Attributes: {attributes}")
        
        return jsonify({
            "response": "registered",
            "facility_id": facility_id,
            "secret_key": secret_key,
            "public_key": public_key
        })
        
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({"error": "Registration failed"}), 500

@api.route('/distribute_global_model', methods=['POST'])
def distribute_global_model():
    """Distribute encrypted global model to authorized facilities"""
    try:
        distribution_data = request.get_json()
        
        encrypted_model = distribution_data['encrypted_model']
        round_num = distribution_data['round']
        
        print(f"Distributing global model for round {round_num}")
        
        # Define access policy (can be customized)
        access_policy = {
            'facility_type': 'hospital',
            'certified': True
        }
        
        successful_distributions = 0
        
        # Distribute to all authorized facilities
        for facility_id, facility_info in registered_facilities.items():
            facility_attributes = facility_info['attributes']
            
            # Check if facility satisfies access policy
            if check_facility_attributes(facility_attributes, access_policy):
                # Re-encrypt with facility-specific policy
                facility_encrypted_model = encrypt_model_with_cp_abe(
                    encrypted_model['encrypted_data'], 
                    access_policy
                )
                
                # Add facility-specific information
                facility_encrypted_model['facility_id'] = facility_id
                facility_encrypted_model['round'] = round_num
                facility_encrypted_model['distribution_timestamp'] = time.time()
                
                # Distribute to facility
                if distribute_to_facility(facility_id, facility_encrypted_model):
                    successful_distributions += 1
                    
            else:
                print(f"Facility {facility_id} does not satisfy access policy")
        
        print(f"Successfully distributed model to {successful_distributions}/{len(registered_facilities)} facilities")
        
        return jsonify({
            "response": "distributed",
            "successful_distributions": successful_distributions,
            "total_facilities": len(registered_facilities),
            "round": round_num
        })
        
    except Exception as e:
        print(f"Distribution error: {e}")
        return jsonify({"error": "Distribution failed"}), 500

@api.route('/get_public_key', methods=['GET'])
def get_public_key():
    """Provide public key for CP-ABE system"""
    return jsonify({
        "public_key": public_key,
        "attribute_universe": attribute_universe,
        "system_status": "active"
    })

@api.route('/facility_list', methods=['GET'])
def get_facility_list():
    """Get list of registered facilities (admin endpoint)"""
    facility_list = []
    for facility_id, info in registered_facilities.items():
        facility_list.append({
            'facility_id': facility_id,
            'attributes': info['attributes'],
            'registration_timestamp': info['registration_timestamp'],
            'status': info['status']
        })
    
    return jsonify({
        "registered_facilities": facility_list,
        "total_count": len(registered_facilities)
    })

@api.route('/revoke_facility', methods=['POST'])
def revoke_facility():
    """Revoke facility access (admin endpoint)"""
    try:
        revoke_data = request.get_json()
        facility_id = revoke_data['facility_id']
        
        if facility_id in registered_facilities:
            registered_facilities[facility_id]['status'] = 'revoked'
            
            if facility_id in issued_keys:
                del issued_keys[facility_id]
            
            print(f"Facility {facility_id} access revoked")
            return jsonify({"response": "revoked", "facility_id": facility_id})
        else:
            return jsonify({"error": "Facility not found"}), 404
            
    except Exception as e:
        print(f"Revocation error: {e}")
        return jsonify({"error": "Revocation failed"}), 500

if __name__ == '__main__':
    print(f"Starting Hierarchical Federated Learning Trusted Authority")
    
    # Initialize CP-ABE system
    public_key, master_secret_key = setup_cp_abe()
    
    print(f"Trusted Authority Public Key: {public_key[:16]}...")
    print(f"Listening on port: {config.ta_port}")
    print(f"PoW Difficulty: {config.pow_difficulty} leading zeros")
    print(f"Attribute Universe: {attribute_universe}")
    
    # Start the Trusted Authority server
    api.run(host=config.ta_address, 
           port=config.ta_port, 
           debug=False, 
           threaded=True)