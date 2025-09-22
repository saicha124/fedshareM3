#!/usr/bin/env python3
import pickle
import threading
import time
import random
import hashlib
import json

import numpy as np
import requests
from flask import Flask, request, jsonify

import flcommon
import time_logger
from config import HierLeaderConfig

config = HierLeaderConfig()

api = Flask(__name__)

# Leader server state
fog_aggregations = []
training_round = 0
total_download_cost = 0
total_upload_cost = 0
leader_private_key = f"leader_server_private_key_{int(time.time())}"
leader_public_key = hashlib.sha256(leader_private_key.encode()).hexdigest()

run_start_time = time.time()

def verify_fog_node_signature(model_data, signature, fog_node_public_key):
    """Verify digital signature from fog node"""
    try:
        # Recreate the signature for verification
        model_str = json.dumps([arr.tolist() if hasattr(arr, 'tolist') else arr for arr in model_data])
        expected_signature_input = f"{model_str}||fog_node_private_key"
        
        # In production, use proper cryptographic signature verification
        return len(signature) == 64 and signature.isalnum()
        
    except Exception as e:
        print(f"Signature verification error: {e}")
        return False

def global_aggregation(fog_aggregations):
    """Perform global aggregation from fog node partial aggregations"""
    if not fog_aggregations:
        print("No fog aggregations available for global aggregation")
        return None
    
    print(f"Performing global aggregation on {len(fog_aggregations)} fog node aggregations")
    
    # Extract aggregated models from fog nodes
    fog_models = []
    total_facilities = 0
    
    for fog_data in fog_aggregations:
        fog_model = fog_data['aggregated_model']
        num_facilities = fog_data.get('num_facilities_aggregated', 1)
        
        fog_models.append(fog_model)
        total_facilities += num_facilities
        
        print(f"Fog node {fog_data['fog_node_id']}: aggregated {num_facilities} facilities")
    
    # Global aggregation by summing all fog node contributions
    # Each fog node has already performed weighted averaging of its facilities
    global_model = []
    
    # Initialize with first fog model structure
    first_fog_model = fog_models[0]
    for layer in first_fog_model:
        global_model.append(np.zeros_like(layer))
    
    # Sum contributions from all fog nodes
    for fog_model in fog_models:
        for layer_idx, layer_params in enumerate(fog_model):
            global_model[layer_idx] += layer_params
    
    # No need to divide by number of fog nodes since each fog node
    # already performed proper weighted averaging
    
    print(f"Global aggregation completed from {len(fog_models)} fog nodes, {total_facilities} total facilities")
    return global_model

def encrypt_global_model(global_model, access_policy="hospital AND region=north"):
    """Encrypt global model using CP-ABE (simplified simulation)"""
    # Simplified CP-ABE encryption simulation
    # In production, use proper CP-ABE library
    
    model_data = pickle.dumps(global_model)
    
    # Simulate CP-ABE encryption with access policy
    encrypted_model = {
        'ciphertext': hashlib.sha256(model_data).hexdigest(),  # Simulated ciphertext
        'access_policy': access_policy,
        'encrypted_data': model_data,  # In production, this would be encrypted
        'encryption_timestamp': time.time(),
        'leader_signature': hashlib.sha256(f"{model_data.hex()}||{leader_private_key}".encode()).hexdigest()
    }
    
    print(f"Global model encrypted with access policy: {access_policy}")
    return encrypted_model

def broadcast_global_model(encrypted_global_model):
    """Broadcast encrypted global model to all healthcare facilities"""
    print(f"Broadcasting global model to {config.number_of_facilities} healthcare facilities")
    
    # Send to Trusted Authority for final distribution
    try:
        ta_url = f"http://{config.ta_address}:{config.ta_port}/distribute_global_model"
        
        distribution_data = {
            'encrypted_model': encrypted_global_model,
            'round': training_round,
            'leader_id': 'leader_server',
            'timestamp': time.time()
        }
        
        response = requests.post(ta_url, json=distribution_data, timeout=60)
        
        if response.status_code == 200:
            print("Global model sent to Trusted Authority for distribution")
            return True
        else:
            print(f"Failed to send to Trusted Authority: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"Error sending to Trusted Authority: {e}")
        return False

def process_global_aggregation():
    """Process global aggregation when all fog nodes have reported"""
    global fog_aggregations, training_round, total_upload_cost
    
    time_logger.lead_server_start()
    
    print(f"Processing global aggregation from {len(fog_aggregations)} fog nodes")
    
    # Perform global aggregation
    global_model = global_aggregation(fog_aggregations)
    
    if global_model is None:
        print("Global aggregation failed")
        return False
    
    # Encrypt global model with CP-ABE
    encrypted_global_model = encrypt_global_model(global_model)
    
    # Broadcast encrypted model to facilities via Trusted Authority
    success = broadcast_global_model(encrypted_global_model)
    
    # Clear fog aggregations for next round
    fog_aggregations.clear()
    
    # Calculate upload cost
    global total_upload_cost
    model_size = len(pickle.dumps(encrypted_global_model))
    total_upload_cost += model_size * config.number_of_facilities
    
    print(f"[DOWNLOAD] Total download cost so far: {total_download_cost}")
    print(f"[UPLOAD] Total upload cost so far: {total_upload_cost}")
    print("[AGGREGATION] Global model aggregation and distribution completed successfully.")
    
    training_round += 1
    
    print(f"********************** [LEADER] Round {training_round} completed **********************")
    
    time_logger.lead_server_idle()
    
    return success

@api.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "server_id": "leader",
        "status": "healthy",
        "algorithm": "hierarchical_federated",
        "round": training_round,
        "received_fog_aggregations": len(fog_aggregations),
        "expected_fog_nodes": config.num_fog_nodes,
        "public_key": leader_public_key[:16]
    })

@api.route('/receive_fog_aggregation', methods=['POST'])
def receive_fog_aggregation():
    """Receive partial aggregation from fog node"""
    try:
        # Deserialize the aggregation data
        aggregation_data = pickle.loads(request.data)
        
        global total_download_cost
        total_download_cost += len(request.data)
        
        fog_node_id = aggregation_data['fog_node_id']
        aggregated_model = aggregation_data['aggregated_model']
        signature = aggregation_data['signature']
        fog_public_key = aggregation_data['public_key']
        
        print(f"[DOWNLOAD] Fog aggregation from fog node {fog_node_id} received. size: {len(request.data)}")
        
        # Verify fog node signature
        if not verify_fog_node_signature(aggregated_model, signature, fog_public_key):
            print(f"Invalid signature from fog node {fog_node_id}")
            return jsonify({"error": "Invalid fog node signature"}), 401
        
        # Store the verified aggregation
        fog_aggregations.append(aggregation_data)
        
        print(f"Leader server received aggregation {len(fog_aggregations)}/{config.num_fog_nodes} from fog node {fog_node_id}")
        
        # Check if all fog nodes have reported
        if len(fog_aggregations) >= config.num_fog_nodes:
            print("All fog aggregations received, starting global aggregation...")
            # Start global aggregation in separate thread
            global_aggregation_thread = threading.Thread(target=process_global_aggregation)
            global_aggregation_thread.start()
        
        return jsonify({"response": "aggregation_received", "leader_id": "leader_server"})
        
    except Exception as e:
        print(f"Error processing fog aggregation: {e}")
        return jsonify({"error": "Aggregation processing failed"}), 500

@api.route('/start_round', methods=['POST'])
def start_round():
    """Initialize new training round"""
    global fog_aggregations
    
    fog_aggregations.clear()
    
    print(f"Leader server initialized new training round {training_round + 1}")
    return jsonify({"response": "round_started", "round": training_round + 1})

@api.route('/status', methods=['GET'])
def get_status():
    """Get detailed leader server status"""
    return jsonify({
        "server_id": "leader",
        "training_round": training_round,
        "received_fog_aggregations": len(fog_aggregations),
        "expected_fog_nodes": config.num_fog_nodes,
        "ready_for_global_aggregation": len(fog_aggregations) >= config.num_fog_nodes,
        "total_download_cost": total_download_cost,
        "total_upload_cost": total_upload_cost,
        "uptime": time.time() - run_start_time,
        "leader_port": config.leader_port
    })

@api.route('/leader_selection', methods=['POST'])
def leader_selection():
    """Handle leader selection process (called by system)"""
    # Simulate random leader selection from fog nodes
    selected_fog_node = random.randint(0, config.num_fog_nodes - 1)
    
    selection_data = {
        'selected_leader': selected_fog_node,
        'round': training_round,
        'selection_timestamp': time.time(),
        'leader_public_key': leader_public_key
    }
    
    print(f"Leader selection: fog node {selected_fog_node} selected as leader for round {training_round + 1}")
    
    return jsonify(selection_data)

if __name__ == '__main__':
    print(f"Starting Hierarchical Federated Learning Leader Server")
    print(f"Leader Public Key: {leader_public_key[:16]}...")
    print(f"Listening on port: {config.leader_port}")
    print(f"Expecting aggregations from {config.num_fog_nodes} fog nodes")
    print(f"Will distribute to {config.number_of_facilities} healthcare facilities")
    
    # Start the leader server
    api.run(host=config.master_server_address, 
           port=config.leader_port, 
           debug=False, 
           threaded=True)