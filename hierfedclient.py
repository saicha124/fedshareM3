#!/usr/bin/env python3
import pickle
import sys
import threading
import hashlib
import time
import secrets
import json

import numpy as np
import requests
from flask import Flask, request
from requests_toolbelt.adapters import source
import tensorflow as tf

import flcommon
import mnistcommon
import time_logger
from config import HierFacilityConfig

# Set deterministic seeds for consistent initialization across all facilities
np.random.seed(42)
tf.random.set_seed(42)

config = HierFacilityConfig(int(sys.argv[1]))

facility_datasets = mnistcommon.load_train_dataset(config.number_of_facilities, permute=True)

api = Flask(__name__)

round_weight = 0
training_round = 0
total_upload_cost = 0
total_download_cost = 0
facility_private_key = secrets.token_hex(32)  # Simulated private key
facility_public_key = hashlib.sha256(facility_private_key.encode()).hexdigest()

# Proof-of-Work for Sybil Resistance
def solve_proof_of_work(facility_id, target_difficulty):
    """Solve Proof-of-Work challenge to prevent Sybil attacks"""
    nonce = 0
    facility_data = f"{facility_id}||{facility_public_key}"
    
    while True:
        challenge_input = f"{nonce}||{facility_data}"
        hash_result = hashlib.sha256(challenge_input.encode()).hexdigest()
        
        # Check if hash has required number of leading zeros
        if int(hash_result, 16) < config.pow_target:
            print(f"Facility {facility_id} solved PoW challenge with nonce: {nonce}")
            return nonce, hash_result
        
        nonce += 1
        if nonce % 10000 == 0:
            print(f"Facility {facility_id} PoW attempt: {nonce}")

# Differential Privacy - Add Gaussian noise to model parameters
def add_differential_privacy(model_weights, noise_scale):
    """Add Gaussian noise for differential privacy"""
    noisy_weights = []
    for layer_weights in model_weights:
        noise = np.random.normal(0, noise_scale, layer_weights.shape)
        noisy_layer = layer_weights + noise
        noisy_weights.append(noisy_layer)
    
    print(f"Applied differential privacy with noise scale: {noise_scale}")
    return noisy_weights

# Shamir's Secret Sharing Implementation
def shamirs_secret_sharing(data, num_shares, threshold):
    """Split data into secret shares using Shamir's Secret Sharing"""
    # Simplified implementation for demonstration
    # In production, use proper cryptographic library
    shares = []
    
    # Serialize the data
    data_bytes = pickle.dumps(data)
    data_size = len(data_bytes)
    
    # Create shares by splitting data with polynomial interpolation simulation
    for i in range(num_shares):
        # Generate pseudo-random shares (simplified)
        share_data = {
            'share_id': i + 1,
            'data_fragment': data_bytes[i::num_shares] if i < len(data_bytes) else b'',
            'size_info': data_size,
            'threshold': threshold,
            'total_shares': num_shares
        }
        shares.append(share_data)
    
    print(f"Created {num_shares} secret shares with threshold {threshold}")
    return shares

# Digital Signature (simplified)
def sign_data(data, private_key):
    """Create digital signature for data"""
    data_str = json.dumps(data, sort_keys=True, default=str)
    signature_input = f"{data_str}||{private_key}"
    signature = hashlib.sha256(signature_input.encode()).hexdigest()
    return signature

def send_to_validator_committee(share_data, share_index):
    """Send secret share to validator committee for verification"""
    # Create signed share
    signed_share = {
        'facility_id': config.facility_index,
        'share': share_data,
        'signature': sign_data(share_data, facility_private_key),
        'public_key': facility_public_key,
        'round': training_round,
        'timestamp': time.time()
    }
    
    # Send to validator committee (round-robin or all validators)
    validator_index = share_index % config.committee_size
    validator_port = config.committee_base_port + validator_index
    
    url = f'http://{config.server_address}:{validator_port}/validate_share'
    
    try:
        response = requests.post(url, json=signed_share, timeout=30)
        if response.status_code == 200:
            print(f"Share {share_index} sent to validator {validator_index} successfully")
            return True
        else:
            print(f"Failed to send share to validator {validator_index}: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Network error sending to validator {validator_index}: {e}")
        return False

def start_next_round(data):
    """Start next training round with hierarchical federated learning"""
    time_logger.client_start()
    
    x_train, y_train = facility_datasets[config.facility_index][0], facility_datasets[config.facility_index][1]
    
    model = mnistcommon.get_model()
    global training_round, round_weight
    
    if training_round != 0:
        round_weight = pickle.loads(data)
        model.set_weights(round_weight)
    
    print(f"Model: HierarchicalFederated, "
          f"Round: {training_round + 1}/{config.hier_training_rounds}, "
          f"Facility {config.facility_index + 1}/{config.number_of_facilities}, "
          f"Dataset Size: {len(x_train)}")
    
    # Local training
    model.fit(x_train, y_train, 
             epochs=config.hier_epochs, 
             batch_size=config.hier_batch_size, 
             verbose=config.verbose,
             validation_split=config.validation_split)
    
    # Evaluate local facility performance
    x_test, y_test = mnistcommon.load_test_dataset()
    local_results = model.evaluate(x_test, y_test, verbose=0)
    local_loss = local_results[0]
    local_accuracy = local_results[1]
    
    print(f"Facility {config.facility_index} Local Performance:")
    print(f"  loss: {local_loss:.6f}")
    print(f"  accuracy: {local_accuracy:.6f}")
    
    # Get model weights for sharing
    model_weights = model.get_weights()
    
    # Apply differential privacy
    dp_weights = add_differential_privacy(model_weights, config.dp_noise_scale)
    
    # Create secret shares using Shamir's Secret Sharing
    secret_shares = shamirs_secret_sharing(dp_weights, config.num_fog_nodes, config.secret_threshold)
    
    # Send shares to validator committee for verification
    print(f"Sending {len(secret_shares)} secret shares to validator committee...")
    
    successful_sends = 0
    for i, share in enumerate(secret_shares):
        if send_to_validator_committee(share, i):
            successful_sends += 1
    
    print(f"Successfully sent {successful_sends}/{len(secret_shares)} shares to validators")
    
    # Update round counter
    training_round += 1
    
    # Log completion
    print(f"********************** [FACILITY] Round {training_round} completed **********************")
    
    time_logger.client_idle()

@api.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {
        "facility_id": config.facility_index,
        "status": "healthy",
        "algorithm": "hierarchical_federated",
        "round": training_round,
        "public_key": facility_public_key
    }

@api.route('/start_round', methods=['POST'])
def start_round():
    """Start training round"""
    my_thread = threading.Thread(target=start_next_round, args=(request.data,))
    my_thread.start()
    return {"response": "ok", "facility_id": config.facility_index}

@api.route('/register', methods=['POST'])
def register_facility():
    """Register facility with Trusted Authority using PoW"""
    print(f"Registering facility {config.facility_index} with Trusted Authority...")
    
    # Solve Proof-of-Work challenge
    nonce, hash_result = solve_proof_of_work(config.facility_index, config.pow_difficulty)
    
    registration_data = {
        'facility_id': config.facility_index,
        'public_key': facility_public_key,
        'nonce': nonce,
        'hash_result': hash_result,
        'attributes': {
            'facility_type': 'hospital',
            'region': 'north',
            'certified': True
        }
    }
    
    # Send registration to Trusted Authority
    try:
        ta_url = f'http://{config.ta_address}:{config.ta_port}/register_facility'
        response = requests.post(ta_url, json=registration_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Facility {config.facility_index} registered successfully")
            return {"response": "registered", "facility_id": config.facility_index, "secret_key": result.get('secret_key')}
        else:
            print(f"Registration failed: {response.status_code}")
            return {"response": "registration_failed"}, response.status_code
            
    except requests.RequestException as e:
        print(f"Registration error: {e}")
        return {"response": "registration_error"}, 500

if __name__ == '__main__':
    print(f"Starting Hierarchical Federated Learning Facility {config.facility_index}")
    print(f"Facility Public Key: {facility_public_key[:16]}...")
    print(f"Listening on port: {config.facility_port}")
    
    # Start the facility server
    api.run(host=config.client_address, 
           port=config.facility_port, 
           debug=False, 
           threaded=True)