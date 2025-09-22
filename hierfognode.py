#!/usr/bin/env python3
import pickle
import sys
import threading
import hashlib
import time
import json

import numpy as np
import requests
from flask import Flask, request, jsonify

import time_logger
from config import HierFogNodeConfig

config = HierFogNodeConfig(int(sys.argv[1]))

api = Flask(__name__)

training_round = 0
received_shares = []
total_download_cost = 0
total_upload_cost = 0

# Fog node's cryptographic keys (simplified)
fog_node_private_key = f"fog_node_{config.fog_node_index}_private_key"
fog_node_public_key = hashlib.sha256(fog_node_private_key.encode()).hexdigest()

def verify_committee_signature(data, signature, committee_public_key):
    """Verify digital signature from validator committee"""
    # Simplified signature verification
    data_str = json.dumps(data, sort_keys=True, default=str)
    expected_signature = hashlib.sha256(f"{data_str}||committee_signature".encode()).hexdigest()
    
    # In production, use proper cryptographic signature verification
    return len(signature) == 64 and signature.isalnum()  # Basic format check

def reconstruct_secret_shares(shares):
    """Reconstruct model parameters from secret shares (simplified Shamir's)"""
    if len(shares) < config.secret_threshold:
        print(f"Insufficient shares: {len(shares)}, need at least {config.secret_threshold}")
        return None
    
    # Simplified reconstruction - in production use proper Shamir's Secret Sharing
    reconstructed_data = {}
    
    for i, share_info in enumerate(shares):
        share_data = share_info['share']
        facility_id = share_info['facility_id']
        
        # Reconstruct by combining share fragments
        if 'data_fragment' in share_data:
            if facility_id not in reconstructed_data:
                reconstructed_data[facility_id] = b''
            reconstructed_data[facility_id] += share_data['data_fragment']
    
    # Deserialize reconstructed model parameters for each facility
    facility_models = {}
    for facility_id, data_bytes in reconstructed_data.items():
        try:
            if data_bytes:
                model_params = pickle.loads(data_bytes)
                facility_models[facility_id] = model_params
        except Exception as e:
            print(f"Error reconstructing facility {facility_id} data: {e}")
            continue
    
    print(f"Reconstructed models from {len(facility_models)} facilities")
    return facility_models

def fedavg_aggregation(facility_models):
    """Perform FedAvg aggregation on reconstructed model parameters"""
    if not facility_models:
        print("No facility models available for aggregation")
        return None
    
    print(f"Performing FedAvg aggregation on {len(facility_models)} facility models")
    
    # Get the structure from the first model
    first_facility_id = list(facility_models.keys())[0]
    first_model = facility_models[first_facility_id]
    
    # Initialize aggregated model with zeros
    aggregated_model = []
    for layer in first_model:
        aggregated_model.append(np.zeros_like(layer))
    
    # Sum all model parameters
    num_facilities = len(facility_models)
    for facility_id, model_params in facility_models.items():
        for layer_idx, layer_params in enumerate(model_params):
            aggregated_model[layer_idx] += layer_params / num_facilities
    
    print(f"FedAvg aggregation completed using {num_facilities} facilities")
    return aggregated_model

def sign_aggregated_model(model_data):
    """Create digital signature for aggregated model"""
    model_str = json.dumps([arr.tolist() if hasattr(arr, 'tolist') else arr for arr in model_data])
    signature_input = f"{model_str}||{fog_node_private_key}"
    signature = hashlib.sha256(signature_input.encode()).hexdigest()
    return signature

def send_to_leader_server(aggregated_model):
    """Send aggregated model to leader server"""
    # Create signed aggregated model
    signed_model = {
        'fog_node_id': config.fog_node_index,
        'aggregated_model': aggregated_model,
        'signature': sign_aggregated_model(aggregated_model),
        'public_key': fog_node_public_key,
        'round': training_round,
        'timestamp': time.time(),
        'num_facilities_aggregated': len(received_shares)
    }
    
    try:
        leader_url = f'http://{config.server_address}:{config.leader_port}/receive_fog_aggregation'
        
        # Serialize the model data for transmission
        serialized_model = pickle.dumps(signed_model)
        
        response = requests.post(leader_url, data=serialized_model, 
                               headers={'Content-Type': 'application/octet-stream'},
                               timeout=60)
        
        global total_upload_cost
        total_upload_cost += len(serialized_model)
        
        if response.status_code == 200:
            print(f"Fog node {config.fog_node_index} successfully sent aggregation to leader server")
            print(f"[UPLOAD] Sent aggregated model to leader, size: {len(serialized_model)}")
            return True
        else:
            print(f"Failed to send to leader server: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"Network error sending to leader server: {e}")
        return False

def process_aggregation():
    """Process the aggregation when enough shares are received"""
    global received_shares, training_round
    
    time_logger.server_start()
    
    print(f"Processing aggregation for fog node {config.fog_node_index}")
    print(f"Received {len(received_shares)} shares from facilities")
    
    # Reconstruct model parameters from secret shares
    facility_models = reconstruct_secret_shares(received_shares)
    
    if not facility_models:
        print("Failed to reconstruct facility models from shares")
        return False
    
    # Perform FedAvg aggregation
    aggregated_model = fedavg_aggregation(facility_models)
    
    if aggregated_model is None:
        print("FedAvg aggregation failed")
        return False
    
    # Send aggregated model to leader server
    success = send_to_leader_server(aggregated_model)
    
    # Clear received shares for next round
    received_shares.clear()
    training_round += 1
    
    print(f"[DOWNLOAD] Total download cost so far: {total_download_cost}")
    print(f"[UPLOAD] Total upload cost so far: {total_upload_cost}")
    print(f"********************** [FOG NODE] Round {training_round} completed **********************")
    
    time_logger.server_idle()
    
    return success

@api.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "fog_node_id": config.fog_node_index,
        "status": "healthy",
        "algorithm": "hierarchical_federated",
        "round": training_round,
        "received_shares": len(received_shares),
        "public_key": fog_node_public_key[:16]
    })

@api.route('/receive_share', methods=['POST'])
def receive_share():
    """Receive verified secret share from validator committee"""
    try:
        share_data = request.get_json()
        
        global total_download_cost
        total_download_cost += len(request.data)
        
        print(f"[DOWNLOAD] Share received from facility {share_data.get('facility_id')}, size: {len(request.data)}")
        
        # Verify committee signature (simplified)
        committee_signature = share_data.get('committee_signature', '')
        if not verify_committee_signature(share_data.get('share', {}), committee_signature, 'committee_key'):
            print(f"Invalid committee signature for share from facility {share_data.get('facility_id')}")
            return jsonify({"error": "Invalid committee signature"}), 401
        
        # Store the verified share
        received_shares.append(share_data)
        
        print(f"Fog node {config.fog_node_index} received share {len(received_shares)}/{config.number_of_facilities}")
        
        # Check if we have enough shares to start aggregation
        if len(received_shares) >= config.number_of_facilities:
            print(f"All shares received, starting aggregation...")
            # Start aggregation in separate thread
            aggregation_thread = threading.Thread(target=process_aggregation)
            aggregation_thread.start()
        
        return jsonify({"response": "share_received", "fog_node_id": config.fog_node_index})
        
    except Exception as e:
        print(f"Error processing share: {e}")
        return jsonify({"error": "Share processing failed"}), 500

@api.route('/reset_round', methods=['POST'])
def reset_round():
    """Reset for new training round"""
    global received_shares, training_round
    
    received_shares.clear()
    print(f"Fog node {config.fog_node_index} reset for new round")
    
    return jsonify({"response": "reset_complete", "fog_node_id": config.fog_node_index})

@api.route('/status', methods=['GET'])
def get_status():
    """Get detailed fog node status"""
    return jsonify({
        "fog_node_id": config.fog_node_index,
        "training_round": training_round,
        "received_shares": len(received_shares),
        "expected_shares": config.number_of_facilities,
        "ready_for_aggregation": len(received_shares) >= config.number_of_facilities,
        "total_download_cost": total_download_cost,
        "total_upload_cost": total_upload_cost,
        "fog_node_port": config.fog_node_port
    })

if __name__ == '__main__':
    print(f"Starting Hierarchical Federated Learning Fog Node {config.fog_node_index}")
    print(f"Fog Node Public Key: {fog_node_public_key[:16]}...")
    print(f"Listening on port: {config.fog_node_port}")
    print(f"Expected shares from {config.number_of_facilities} facilities")
    
    # Start the fog node server
    api.run(host=config.server_address, 
           port=config.fog_node_port, 
           debug=False, 
           threaded=True)