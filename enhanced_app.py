#!/usr/bin/env python3
import http.server
import socketserver
import subprocess
import urllib.parse
import os
import threading
import json
import time
import re
from datetime import datetime

PORT = 5000

# Track running processes and their progress
running_processes = {}
progress_data = {}

def parse_logs_for_progress(algorithm):
    """Parse log files to extract training progress"""
    # Import and reload config to get current values
    import importlib
    import config
    importlib.reload(config)
    
    # Get current configuration values
    total_clients = config.Config.number_of_clients
    total_rounds = config.Config.training_rounds
    num_servers = config.Config.num_servers
    
    # Generate dynamic log directory names based on current config
    if algorithm == 'fedavg':
        log_dir_name = f"fedavg-mnist-client-{total_clients}"
    elif algorithm == 'hierfed':
        # Hierarchical federated learning has different structure
        from config import HierConfig
        hier_config = HierConfig()
        facilities = hier_config.number_of_facilities
        fog_nodes = hier_config.num_fog_nodes
        validators = hier_config.committee_size
        log_dir_name = f"hierfed-facilities-{facilities}-fog-{fog_nodes}-validators-{validators}"
    else:
        log_dir_name = f"{algorithm}-mnist-client-{total_clients}-server-{num_servers}"
    
    log_dir = f"logs/{log_dir_name}"
    
    if algorithm not in ['fedshare', 'fedavg', 'scotch', 'hierfed']:
        return {}
        
    progress = {
        'clients_started': 0,
        'total_clients': total_clients,
        'current_round': 0,
        'total_rounds': total_rounds,
        'training_progress': 0,
        'status': 'not_started',
        'results': [],
        'metrics': {}
    }
    
    if not os.path.exists(log_dir):
        return progress
    
    # Handle hierarchical federated learning structure differently
    if algorithm == 'hierfed':
        from config import HierConfig
        hier_config = HierConfig()
        total_clients = hier_config.number_of_facilities
        total_rounds = hier_config.hier_training_rounds
        progress['total_clients'] = total_clients
        progress['total_rounds'] = total_rounds
        
        # Check healthcare facility logs
        for i in range(total_clients):
            client_log = f"{log_dir}/hierfedclient-{i}.log"
            if os.path.exists(client_log):
                progress['clients_started'] += 1
                try:
                    with open(client_log, 'r') as f:
                        content = f.read()
                        
                    # Extract round information for hierarchical structure
                    rounds = re.findall(r'Round: (\d+)/(\d+)', content)
                    if rounds:
                        latest_round = max([int(r[0]) for r in rounds])
                        progress['current_round'] = max(progress['current_round'], latest_round)
                    
                    # Extract facility completion
                    completed_rounds = content.count('FACILITY] Round')
                    training_finished = content.count('Training completed')
                    
                    if training_finished > 0:
                        progress['training_progress'] = 100
                    else:
                        round_progress = min(100, (completed_rounds / max(1, total_rounds)) * 100) if total_rounds > 0 else 0
                        progress['training_progress'] = max(progress['training_progress'], round_progress)
                    
                    # Extract performance metrics
                    accuracy_matches = re.findall(r'accuracy: ([\d.]+)', content)
                    loss_matches = re.findall(r'loss: ([\d.]+)', content)
                    if accuracy_matches:
                        progress['metrics'][f'facility_{i}_accuracy'] = float(accuracy_matches[-1])
                    if loss_matches:
                        progress['metrics'][f'facility_{i}_loss'] = float(loss_matches[-1])
                        
                except Exception as e:
                    print(f"Error reading facility log {client_log}: {e}")
        
        # Check fog node and leader server logs for global completion
        leader_log = f"{log_dir}/hierleadserver.log"
        if os.path.exists(leader_log):
            try:
                with open(leader_log, 'r') as f:
                    content = f.read()
                
                global_aggregations = content.count('LEADER] Round')
                if global_aggregations >= total_rounds:
                    progress['training_progress'] = 100
                    progress['status'] = 'completed'
                    
            except Exception as e:
                print(f"Error reading leader server log: {e}")
    else:
        # Check client logs for training progress (original logic)
        for i in range(total_clients):
            client_log = f"{log_dir}/{algorithm}client-{i}.log"
            if os.path.exists(client_log):
                progress['clients_started'] += 1
            try:
                with open(client_log, 'r') as f:
                    content = f.read()
                    
                # Extract round information
                rounds = re.findall(r'Round: (\d+)/(\d+)', content)
                if rounds:
                    latest_round = max([int(r[0]) for r in rounds])
                    progress['current_round'] = max(progress['current_round'], latest_round)
                
                # Extract training completion
                completed_rounds = content.count('completed')
                training_finished = content.count('Training finished')
                
                # If training is finished, set to 100%, otherwise calculate based on actual total rounds
                if training_finished > 0:
                    progress['training_progress'] = 100
                else:
                    # Calculate percentage based on actual total rounds
                    round_progress = min(100, (completed_rounds / max(1, total_rounds)) * 100) if total_rounds > 0 else 0
                    progress['training_progress'] = max(progress['training_progress'], round_progress)
                
                # Extract accuracy/loss if available
                accuracy_matches = re.findall(r'accuracy: ([\d.]+)', content)
                loss_matches = re.findall(r'loss: ([\d.]+)', content)
                if accuracy_matches:
                    progress['metrics'][f'client_{i}_accuracy'] = float(accuracy_matches[-1])
                if loss_matches:
                    progress['metrics'][f'client_{i}_loss'] = float(loss_matches[-1])
                
                # Extract global performance metrics if available
                global_loss_matches = re.findall(r'📊 Global Test Loss:\s+([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', content)
                global_accuracy_matches = re.findall(r'🎯 Global Test Accuracy:\s+([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', content)
                if global_loss_matches:
                    progress['metrics']['global_loss'] = float(global_loss_matches[-1])
                if global_accuracy_matches:
                    progress['metrics']['global_accuracy'] = float(global_accuracy_matches[-1])
                    
            except Exception as e:
                print(f"Error reading client log {client_log}: {e}")
    
    # Check server logs for completion
    server_log = f"{log_dir}/{algorithm}server.log" if algorithm == 'fedavg' else f"{log_dir}/{algorithm}server-0.log"
    if os.path.exists(server_log):
        try:
            with open(server_log, 'r') as f:
                content = f.read()
                
            # Check for final round completion
            final_round_completed = f"Round {progress['total_rounds']} completed" in content
            if final_round_completed:
                progress['training_progress'] = 100
            else:
                # Extract server aggregation info - calculate based on actual total rounds
                aggregations = content.count('Round completed')
                aggregation_progress = min(100, (aggregations / max(1, total_rounds)) * 100) if total_rounds > 0 else 0
                progress['training_progress'] = max(progress['training_progress'], aggregation_progress)
            
            # Extract global performance metrics from server logs
            global_loss_matches = re.findall(r'📊 Global Test Loss:\s+([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', content)
            global_accuracy_matches = re.findall(r'🎯 Global Test Accuracy:\s+([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', content)
            if global_loss_matches:
                progress['metrics']['global_loss'] = float(global_loss_matches[-1])
            if global_accuracy_matches:
                progress['metrics']['global_accuracy'] = float(global_accuracy_matches[-1])
                
        except Exception as e:
            print(f"Error reading server log: {e}")
    
    # Check lead server for completion
    lead_server_log = f"{log_dir}/{algorithm}leadserver.log"
    if os.path.exists(lead_server_log):
        try:
            with open(lead_server_log, 'r') as f:
                content = f.read()
                
            # Check for successful aggregation completion
            if 'Model aggregation completed successfully' in content:
                progress['training_progress'] = 100
            
            # Extract global performance metrics from lead server logs
            global_loss_matches = re.findall(r'📊 Global Test Loss:\s+([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', content)
            global_accuracy_matches = re.findall(r'🎯 Global Test Accuracy:\s+([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', content)
            if global_loss_matches:
                progress['metrics']['global_loss'] = float(global_loss_matches[-1])
            if global_accuracy_matches:
                progress['metrics']['global_accuracy'] = float(global_accuracy_matches[-1])
                
        except Exception as e:
            print(f"Error reading lead server log: {e}")
    
    # Determine overall status - check completion FIRST
    if progress['clients_started'] == 0:
        progress['status'] = 'not_started'
    elif progress['training_progress'] >= 100:
        progress['status'] = 'completed'
    elif progress['clients_started'] < progress['total_clients']:
        progress['status'] = 'starting_clients'
    elif progress['current_round'] < progress['total_rounds']:
        progress['status'] = 'training'
    else:
        progress['status'] = 'training'
    
    return progress

class EnhancedFedShareHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/config':
            self.update_config()
        else:
            self.send_error(404, "Not Found")
    
    def do_GET(self):
        if self.path == '/':
            self.serve_homepage()
        elif self.path == '/favicon.ico':
            self.send_response(204)  # No Content
            self.end_headers()
        elif self.path == '/reinitialize':
            self.reinitialize_all()
        elif self.path.startswith('/run/'):
            algorithm = self.path.split('/')[-1]
            self.run_algorithm(algorithm)
        elif self.path.startswith('/progress/'):
            algorithm = self.path.split('/')[-1]
            self.get_progress(algorithm)
        elif self.path.startswith('/logs/'):
            algorithm = self.path.split('/')[-1]
            self.show_logs(algorithm)
        elif self.path.startswith('/status/'):
            algorithm = self.path.split('/')[-1]
            self.get_status(algorithm)
        elif self.path == '/current_config':
            self.get_current_config()
        else:
            super().do_GET()
    
    def serve_homepage(self):
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FedShare - Enhanced Federated Learning Framework</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 1.2em;
        }
        .algorithm-section {
            margin: 25px 0;
            padding: 25px;
            border: 2px solid #ecf0f1;
            border-radius: 12px;
            background: linear-gradient(145deg, #f8f9fa, #ffffff);
            transition: all 0.3s ease;
        }
        .algorithm-section:hover {
            border-color: #3498db;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.1);
        }
        .algorithm-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c3e50;
            display: flex;
            align-items: center;
        }
        .algorithm-title .emoji {
            margin-right: 10px;
            font-size: 24px;
        }
        .algorithm-description {
            color: #666;
            margin-bottom: 20px;
            line-height: 1.6;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .btn {
            background: linear-gradient(145deg, #3498db, #2980b9);
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s ease;
            min-width: 120px;
            text-align: center;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
        }
        .btn-success {
            background: linear-gradient(145deg, #27ae60, #219a52);
        }
        .btn-running {
            background: linear-gradient(145deg, #e67e22, #d35400);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .progress-container {
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            display: none;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            border-radius: 10px;
            width: 0%;
            transition: width 0.5s ease;
            position: relative;
        }
        .progress-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-weight: bold;
            font-size: 12px;
        }
        .status-info {
            margin: 10px 0;
            padding: 10px;
            border-radius: 6px;
            font-size: 14px;
        }
        .status-running {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        .status-completed {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .status-error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .metric-item {
            background: white;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
            text-align: center;
        }
        .metric-label {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }
        
        /* Global metrics styling */
        .global-metrics-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            border: 2px solid rgba(255, 255, 255, 0.1);
        }
        
        .global-metrics-title {
            font-size: 24px;
            font-weight: bold;
            margin: 0 0 15px 0;
            text-align: center;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .global-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
        }
        
        .global-metric-item {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .global-metric-item:hover {
            background: rgba(255, 255, 255, 0.25);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        }
        
        .global-metric-label {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 10px;
            opacity: 0.9;
        }
        
        .global-metric-value {
            font-size: 28px;
            font-weight: bold;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .client-metrics-section {
            margin-top: 15px;
        }
        
        .client-metrics-title {
            font-size: 18px;
            color: #2c3e50;
            margin: 0 0 15px 0;
            text-align: center;
            opacity: 0.8;
        }
        
        .client-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        .info-box {
            background: linear-gradient(145deg, #e8f4fd, #ffffff);
            border-left: 4px solid #3498db;
            padding: 20px;
            margin: 25px 0;
            border-radius: 8px;
        }
        .optimization-note {
            background: linear-gradient(145deg, #fff7e6, #ffffff);
            border-left: 4px solid #f39c12;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
        }
        .refresh-btn {
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(145deg, #9b59b6, #8e44ad);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(155, 89, 182, 0.3);
        }
        .reinit-btn {
            position: fixed;
            top: 20px;
            right: 140px;
            background: linear-gradient(145deg, #e74c3c, #c0392b);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
        }
        .reinit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(231, 76, 60, 0.5);
        }
        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }
        .config-item {
            display: flex;
            flex-direction: column;
        }
        .config-item label {
            font-weight: bold;
            margin-bottom: 5px;
            color: #2c3e50;
        }
        .config-item input {
            padding: 8px 12px;
            border: 2px solid #ecf0f1;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }
        .config-item input:focus {
            outline: none;
            border-color: #3498db;
        }
        .config-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 10px;
            border-radius: 6px;
            margin-top: 10px;
        }
        .config-error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 10px;
            border-radius: 6px;
            margin-top: 10px;
        }
    </style>
    <script>
        let updateIntervals = {};
        let completionCounters = {};
        
        function runAlgorithm(algorithm) {
            const button = document.getElementById(algorithm + '-run-btn');
            const progressContainer = document.getElementById(algorithm + '-progress');
            
            button.style.background = 'linear-gradient(145deg, #e67e22, #d35400)';
            button.textContent = 'Starting...';
            button.disabled = true;
            
            fetch('/run/' + algorithm)
                .then(response => response.text())
                .then(data => {
                    progressContainer.style.display = 'block';
                    startProgressTracking(algorithm);
                })
                .catch(error => {
                    console.error('Error:', error);
                    button.textContent = 'Error';
                    button.style.background = 'linear-gradient(145deg, #c0392b, #a93226)';
                });
        }
        
        function startProgressTracking(algorithm) {
            if (updateIntervals[algorithm]) {
                clearInterval(updateIntervals[algorithm]);
            }
            
            updateIntervals[algorithm] = setInterval(() => {
                updateProgress(algorithm);
            }, 2000);
            
            updateProgress(algorithm);
        }
        
        function updateProgress(algorithm) {
            fetch('/progress/' + algorithm)
                .then(response => response.json())
                .then(data => {
                    updateProgressUI(algorithm, data);
                })
                .catch(error => console.error('Progress update error:', error));
        }
        
        function updateProgressUI(algorithm, data) {
            const progressFill = document.getElementById(algorithm + '-progress-fill');
            const progressText = document.getElementById(algorithm + '-progress-text');
            const statusInfo = document.getElementById(algorithm + '-status');
            const metricsContainer = document.getElementById(algorithm + '-metrics');
            const runBtn = document.getElementById(algorithm + '-run-btn');
            
            // Update progress bar - if completed, always show 100%
            let totalProgress;
            if (data.status === 'completed') {
                totalProgress = 100;
            } else {
                totalProgress = Math.min(100, 
                    (data.clients_started / data.total_clients * 25) +
                    (data.current_round / data.total_rounds * 50) +
                    (data.training_progress * 0.25)
                );
            }
            
            progressFill.style.width = totalProgress + '%';
            progressText.textContent = Math.round(totalProgress) + '%';
            
            // Update status
            let statusMessage = '';
            let statusClass = 'status-running';
            
            switch(data.status) {
                case 'not_started':
                    statusMessage = '⏳ Waiting to start...';
                    break;
                case 'starting_clients':
                    statusMessage = `🚀 Starting clients (${data.clients_started}/${data.total_clients})`;
                    break;
                case 'training':
                    statusMessage = `🧠 Training round ${data.current_round}/${data.total_rounds}`;
                    break;
                case 'completed':
                    statusMessage = '✅ Training completed successfully!';
                    statusClass = 'status-completed';
                    
                    // Initialize completion tracking
                    if (!completionCounters[algorithm]) {
                        completionCounters[algorithm] = 0;
                    }
                    completionCounters[algorithm]++;
                    
                    // Check if global metrics are present
                    const hasGlobalMetrics = data.metrics && (data.metrics.global_loss !== undefined || data.metrics.global_accuracy !== undefined);
                    const maxWaitCycles = 15; // 30 seconds max wait
                    
                    if (hasGlobalMetrics || completionCounters[algorithm] >= maxWaitCycles) {
                        // Global metrics found or timeout reached - stop polling
                        clearInterval(updateIntervals[algorithm]);
                        runBtn.textContent = 'Run ' + algorithm.charAt(0).toUpperCase() + algorithm.slice(1);
                        runBtn.style.background = 'linear-gradient(145deg, #3498db, #2980b9)';
                        runBtn.disabled = false;
                        delete completionCounters[algorithm]; // Clean up counter
                    } else {
                        // Still waiting for final metrics - show finalizing status
                        statusMessage = '🔄 Finalizing and capturing final metrics...';
                        runBtn.textContent = 'Finalizing...';
                        runBtn.style.background = 'linear-gradient(145deg, #f39c12, #e67e22)';
                    }
                    break;
            }
            
            statusInfo.innerHTML = `<div class="${statusClass}">${statusMessage}</div>`;
            
            // Update metrics
            if (Object.keys(data.metrics).length > 0) {
                let metricsHTML = '<div class="metrics">';
                
                // Separate global metrics from client metrics
                const globalMetrics = {};
                const clientMetrics = {};
                
                for (const [key, value] of Object.entries(data.metrics)) {
                    if (key.startsWith('global_')) {
                        globalMetrics[key] = value;
                    } else {
                        clientMetrics[key] = value;
                    }
                }
                
                // Display global metrics prominently if training is completed and global metrics exist
                if (data.status === 'completed' && Object.keys(globalMetrics).length > 0) {
                    metricsHTML += `
                        <div class="global-metrics-section">
                            <h3 class="global-metrics-title">🎯 Final Global Performance</h3>
                            <div class="global-metrics">
                    `;
                    
                    for (const [key, value] of Object.entries(globalMetrics)) {
                        const label = key.replace('global_', '').replace('_', ' ').toUpperCase();
                        const icon = key.includes('accuracy') ? '🎯' : '📊';
                        const percentage = key.includes('accuracy') ? ` (${(value * 100).toFixed(2)}%)` : '';
                        metricsHTML += `
                            <div class="global-metric-item">
                                <div class="global-metric-label">${icon} ${label}</div>
                                <div class="global-metric-value">${value.toFixed(6)}${percentage}</div>
                            </div>
                        `;
                    }
                    
                    metricsHTML += `
                            </div>
                        </div>
                    `;
                }
                
                // Display client metrics
                if (Object.keys(clientMetrics).length > 0) {
                    metricsHTML += '<div class="client-metrics-section">';
                    if (Object.keys(globalMetrics).length > 0 && data.status === 'completed') {
                        metricsHTML += '<h4 class="client-metrics-title">Client Performance Details</h4>';
                    }
                    metricsHTML += '<div class="client-metrics">';
                    
                    for (const [key, value] of Object.entries(clientMetrics)) {
                        const label = key.replace('_', ' ').toUpperCase();
                        metricsHTML += `
                            <div class="metric-item">
                                <div class="metric-label">${label}</div>
                                <div class="metric-value">${typeof value === 'number' ? value.toFixed(4) : value}</div>
                            </div>
                        `;
                    }
                    
                    metricsHTML += '</div></div>';
                }
                
                metricsHTML += '</div>';
                metricsContainer.innerHTML = metricsHTML;
            }
        }
        
        function refreshPage() {
            location.reload();
        }
        
        function reinitializeAll() {
            if (confirm('Are you sure you want to kill all clients and servers and reinitialize everything? This will stop all running processes.')) {
                // Update button to show it's working
                const reinitBtn = document.querySelector('.reinit-btn');
                const originalText = reinitBtn.innerHTML;
                reinitBtn.innerHTML = '⏳ Reinitializing...';
                reinitBtn.disabled = true;
                
                fetch('/reinitialize')
                    .then(response => response.text())
                    .then(data => {
                        alert('All processes killed and system reinitialized successfully!');
                        // Reset all progress displays
                        location.reload();
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('Error during reinitialization: ' + error);
                        reinitBtn.innerHTML = originalText;
                        reinitBtn.disabled = false;
                    });
            }
        }
        
        // Configuration functions
        function updateConfig(event) {
            event.preventDefault();
            const formData = new FormData(event.target);
            const configData = {
                clients: parseInt(formData.get('clients')),
                servers: parseInt(formData.get('servers')),
                rounds: parseInt(formData.get('rounds')),
                batch_size: parseInt(formData.get('batch_size')),
                train_dataset_size: parseInt(formData.get('train_dataset_size')),
                epochs: parseInt(formData.get('epochs'))
            };
            
            const submitBtn = event.target.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '⏳ Updating...';
            submitBtn.disabled = true;
            
            fetch('/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(configData)
            })
            .then(response => response.text())
            .then(data => {
                const statusDiv = document.getElementById('config-status');
                statusDiv.innerHTML = '<div class="config-success">✅ Configuration updated successfully! All algorithms will use the new settings.</div>';
                setTimeout(() => {
                    statusDiv.innerHTML = '';
                }, 5000);
            })
            .catch(error => {
                console.error('Error:', error);
                const statusDiv = document.getElementById('config-status');
                statusDiv.innerHTML = '<div class="config-error">❌ Error updating configuration: ' + error + '</div>';
            })
            .finally(() => {
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
            });
        }
        
        function loadCurrentConfig() {
            fetch('/current_config')
                .then(response => response.json())
                .then(config => {
                    document.getElementById('clients').value = config.number_of_clients;
                    document.getElementById('servers').value = config.num_servers;
                    document.getElementById('rounds').value = config.training_rounds;
                    document.getElementById('batch_size').value = config.batch_size;
                    document.getElementById('train_dataset_size').value = config.train_dataset_size;
                    document.getElementById('epochs').value = config.epochs;
                    
                    const statusDiv = document.getElementById('config-status');
                    statusDiv.innerHTML = '<div class="config-success">📥 Current configuration loaded from config.py</div>';
                    setTimeout(() => {
                        statusDiv.innerHTML = '';
                    }, 3000);
                })
                .catch(error => {
                    console.error('Error loading config:', error);
                    const statusDiv = document.getElementById('config-status');
                    statusDiv.innerHTML = '<div class="config-error">❌ Error loading current configuration</div>';
                });
        }
        
        // Load current config on page load
        window.addEventListener('load', loadCurrentConfig);
        
        // Removed automatic page refresh to prevent interrupting training progress
    </script>
</head>
<body>
    <button class="refresh-btn" onclick="refreshPage()">🔄 Refresh</button>
    <button class="reinit-btn" onclick="reinitializeAll()">🛑 Reinitialize All</button>
    
    <div class="container">
        <h1>🚀 FedShare Framework</h1>
        <p class="subtitle">Enhanced Federated Learning with Real-Time Progress</p>
        
        <div class="optimization-note">
            <strong>⚡ Performance Optimized:</strong> 
            Reduced to 3 clients, 2 rounds, and 6K dataset samples for faster training and demonstration.
            Training typically completes in 2-3 minutes per algorithm.
        </div>
        
        <div class="info-box">
            <strong>🔬 About Federated Learning:</strong> 
            Train machine learning models across distributed clients while preserving privacy. 
            Each algorithm demonstrates different approaches to aggregation and security.
        </div>

        <div class="algorithm-section" style="background: linear-gradient(145deg, #e8f5e8, #ffffff); border-color: #27ae60;">
            <div class="algorithm-title">
                <span class="emoji">⚙️</span>Training Configuration
            </div>
            <div class="algorithm-description">
                Configure training parameters that will be used by all algorithms (FedShare, FedAvg, and SCOTCH).
            </div>
            <form id="config-form" onsubmit="updateConfig(event)">
                <div class="config-grid">
                    <div class="config-item">
                        <label for="clients">Number of Clients:</label>
                        <input type="number" id="clients" name="clients" min="1" max="10" value="3">
                    </div>
                    <div class="config-item">
                        <label for="servers">Number of Servers:</label>
                        <input type="number" id="servers" name="servers" min="1" max="5" value="2">
                    </div>
                    <div class="config-item">
                        <label for="rounds">Training Rounds:</label>
                        <input type="number" id="rounds" name="rounds" min="1" max="10" value="1">
                    </div>
                    <div class="config-item">
                        <label for="batch_size">Batch Size:</label>
                        <input type="number" id="batch_size" name="batch_size" min="1" max="256" value="32">
                    </div>
                    <div class="config-item">
                        <label for="train_dataset_size">Dataset Size:</label>
                        <input type="number" id="train_dataset_size" name="train_dataset_size" min="100" max="60000" value="60000">
                    </div>
                    <div class="config-item">
                        <label for="epochs">Epochs per Round:</label>
                        <input type="number" id="epochs" name="epochs" min="1" max="10" value="1">
                    </div>
                </div>
                <div class="controls" style="margin-top: 20px;">
                    <button type="submit" class="btn">💾 Update Configuration</button>
                    <button type="button" class="btn btn-success" onclick="loadCurrentConfig()">🔄 Load Current</button>
                </div>
            </form>
            <div id="config-status" style="margin-top: 10px;"></div>
        </div>

        <div class="algorithm-section">
            <div class="algorithm-title">
                <span class="emoji">🔐</span>FedShare Algorithm
            </div>
            <div class="algorithm-description">
                Privacy-preserving federated learning with secret sharing techniques.
                Uses cryptographic methods to protect individual client updates during aggregation.
            </div>
            <div class="controls">
                <button id="fedshare-run-btn" class="btn" onclick="runAlgorithm('fedshare')">Run FedShare</button>
                <a href="/logs/fedshare" class="btn btn-success">View Logs</a>
            </div>
            <div id="fedshare-progress" class="progress-container">
                <div class="progress-bar">
                    <div id="fedshare-progress-fill" class="progress-fill">
                        <div id="fedshare-progress-text" class="progress-text">0%</div>
                    </div>
                </div>
                <div id="fedshare-status"></div>
                <div id="fedshare-metrics"></div>
            </div>
        </div>

        <div class="algorithm-section">
            <div class="algorithm-title">
                <span class="emoji">📊</span>FedAvg Algorithm
            </div>
            <div class="algorithm-description">
                Classical federated averaging algorithm. Simple weighted averaging of model parameters
                based on local dataset sizes. The foundational approach for federated learning.
            </div>
            <div class="controls">
                <button id="fedavg-run-btn" class="btn" onclick="runAlgorithm('fedavg')">Run FedAvg</button>
                <a href="/logs/fedavg" class="btn btn-success">View Logs</a>
            </div>
            <div id="fedavg-progress" class="progress-container">
                <div class="progress-bar">
                    <div id="fedavg-progress-fill" class="progress-fill">
                        <div id="fedavg-progress-text" class="progress-text">0%</div>
                    </div>
                </div>
                <div id="fedavg-status"></div>
                <div id="fedavg-metrics"></div>
            </div>
        </div>

        <div class="algorithm-section">
            <div class="algorithm-title">
                <span class="emoji">🎯</span>SCOTCH Algorithm
            </div>
            <div class="algorithm-description">
                Secure aggregation for federated learning with advanced cryptographic guarantees.
                Provides strong privacy protection against both honest-but-curious and malicious adversaries.
            </div>
            <div class="controls">
                <button id="scotch-run-btn" class="btn" onclick="runAlgorithm('scotch')">Run SCOTCH</button>
                <a href="/logs/scotch" class="btn btn-success">View Logs</a>
            </div>
            <div id="scotch-progress" class="progress-container">
                <div class="progress-bar">
                    <div id="scotch-progress-fill" class="progress-fill">
                        <div id="scotch-progress-text" class="progress-text">0%</div>
                    </div>
                </div>
                <div id="scotch-status"></div>
                <div id="scotch-metrics"></div>
            </div>
        </div>

        <div class="algorithm-section" style="background: linear-gradient(145deg, #f0e8ff, #ffffff); border-color: #9b59b6;">
            <div class="algorithm-title">
                <span class="emoji">🌫️</span>Hierarchical Federated Learning
            </div>
            <div class="algorithm-description">
                Advanced hierarchical federated learning with fog nodes, validator committees, and Byzantine fault tolerance.
                Features differential privacy, Shamir's secret sharing, CP-ABE encryption, and Proof-of-Work for Sybil resistance.
                Healthcare facilities → Validator Committee → Fog Nodes → Leader Server → Global Model.
            </div>
            <div class="controls">
                <button id="hierfed-run-btn" class="btn" onclick="runAlgorithm('hierfed')">Run Hierarchical FL</button>
                <a href="/logs/hierfed" class="btn btn-success">View Logs</a>
            </div>
            <div id="hierfed-progress" class="progress-container">
                <div class="progress-bar">
                    <div id="hierfed-progress-fill" class="progress-fill">
                        <div id="hierfed-progress-text" class="progress-text">0%</div>
                    </div>
                </div>
                <div id="hierfed-status"></div>
                <div id="hierfed-metrics"></div>
            </div>
        </div>

        <div class="info-box">
            <strong>📋 Training Configuration:</strong>
            <ul style="margin: 10px 0;">
                <li><strong>Clients:</strong> 3 distributed nodes</li>
                <li><strong>Dataset:</strong> MNIST (6,000 samples total)</li>
                <li><strong>Rounds:</strong> 2 training rounds</li>
                <li><strong>Batch Size:</strong> 32 (optimized for speed)</li>
                <li><strong>Results:</strong> Automatic accuracy and loss tracking</li>
            </ul>
        </div>
    </div>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def get_progress(self, algorithm):
        """Get real-time progress for an algorithm"""
        progress = parse_logs_for_progress(algorithm)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(json.dumps(progress).encode())
    
    def run_algorithm(self, algorithm):
        if algorithm not in ['fedshare', 'fedavg', 'scotch', 'hierfed']:
            self.send_error(400, "Invalid algorithm")
            return
        
        # Kill any existing processes first
        subprocess.run(['pkill', '-f', f'{algorithm}'], capture_output=True)
        time.sleep(1)
        
        # Clean up old logs - generate dynamic log directory names
        import importlib
        import config
        importlib.reload(config)
        
        total_clients = config.Config.number_of_clients
        num_servers = config.Config.num_servers
        
        if algorithm == 'fedavg':
            log_dir_name = f"fedavg-mnist-client-{total_clients}"
        else:
            log_dir_name = f"{algorithm}-mnist-client-{total_clients}-server-{num_servers}"
        
        log_dir_path = f"logs/{log_dir_name}"
        subprocess.run(['rm', '-rf', log_dir_path], capture_output=True)
        os.makedirs(log_dir_path, exist_ok=True)
        
        try:
            if algorithm == 'fedshare':
                # Start FedShare directly managing all processes
                self.start_fedshare_processes(log_dir_path, total_clients, num_servers)
            else:
                # For other algorithms, use the original shell script approach
                script_map = {
                    'fedavg': './start-fedavg.sh', 
                    'scotch': './start-scotch.sh',
                    'hierfed': './start-hierfed.sh'
                }
                script_path = script_map[algorithm]
                print(f"Starting {algorithm}: {script_path}")
                
                process = subprocess.Popen(
                    ['/bin/bash', script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd='.'
                )
                
                running_processes[algorithm] = process
                progress_data[algorithm] = {'status': 'starting', 'start_time': time.time()}
                print(f"Started {algorithm} with PID: {process.pid}")
            
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"{algorithm.upper()} started successfully!".encode())
            
        except Exception as e:
            print(f"Error starting {algorithm}: {str(e)}")
            self.send_error(500, str(e))
    
    def start_fedshare_processes(self, log_dir_path, total_clients, num_servers):
        """Start FedShare processes directly without shell scripts"""
        import socket
        
        # Dictionary to track all spawned processes
        fedshare_processes = {}
        
        print(f"Starting FedShare with {total_clients} clients and {num_servers} servers")
        
        def wait_for_port(host, port, timeout=30):
            """Wait for a port to be available"""
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    if result == 0:
                        return True
                except:
                    pass
                time.sleep(1)
            return False
        
        try:
            # Start logger server
            log_file = open(f"{log_dir_path}/logger_server.log", "w")
            process = subprocess.Popen(
                ['python', '-u', 'logger_server.py'],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd='.',
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            fedshare_processes['logger'] = {'process': process, 'log_file': log_file}
            print(f"Started logger server (PID: {process.pid})")
            
            # Start lead server
            log_file = open(f"{log_dir_path}/fedshareleadserver.log", "w")
            process = subprocess.Popen(
                ['python', '-u', 'fedshareleadserver.py'],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd='.',
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            fedshare_processes['lead'] = {'process': process, 'log_file': log_file}
            print(f"Started lead server (PID: {process.pid})")
            
            # Start regular servers
            for i in range(num_servers):
                log_file = open(f"{log_dir_path}/fedshareserver-{i}.log", "w")
                process = subprocess.Popen(
                    ['python', '-u', 'fedshareserver.py', str(i)],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd='.',
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                )
                fedshare_processes[f'server_{i}'] = {'process': process, 'log_file': log_file}
                print(f"Started server {i} (PID: {process.pid})")
            
            # Wait for servers to be ready
            print("Waiting for servers to initialize...")
            time.sleep(15)  # Give servers time to start
            
            # Start clients
            for i in range(total_clients):
                log_file = open(f"{log_dir_path}/fedshareclient-{i}.log", "w")
                process = subprocess.Popen(
                    ['python', '-u', 'fedshareclient.py', str(i)],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd='.',
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                )
                fedshare_processes[f'client_{i}'] = {'process': process, 'log_file': log_file}
                print(f"Started client {i} (PID: {process.pid})")
            
            # Store all processes in the global running_processes dict
            running_processes['fedshare'] = fedshare_processes
            progress_data['fedshare'] = {'status': 'starting', 'start_time': time.time()}
            
            print("FedShare processes started successfully!")
            
            # Robust startup synchronization with health checks and retry logic
            def initiate_training():
                import time
                import threading
                import requests
                import socket
                from config import Config
                
                def check_client_health(client_id, max_retries=30, delay=2):
                    """Check if client is healthy and ready to receive requests"""
                    port = Config.client_base_port + client_id
                    health_url = f'http://{Config.client_address}:{port}/'
                    
                    for attempt in range(max_retries):
                        try:
                            # First check if port is accessible
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(3)
                            result = sock.connect_ex((Config.client_address, port))
                            sock.close()
                            
                            if result == 0:
                                # Port is open, now check HTTP health
                                response = requests.get(health_url, timeout=5)
                                if response.status_code == 200:
                                    print(f"✅ Client {client_id} is healthy and ready (port {port})")
                                    return True
                                else:
                                    print(f"⚠️ Client {client_id} port {port} accessible but returned {response.status_code}")
                            else:
                                print(f"🔄 Client {client_id} port {port} not ready yet (attempt {attempt + 1}/{max_retries})")
                        except Exception as e:
                            print(f"🔄 Client {client_id} health check failed (attempt {attempt + 1}/{max_retries}): {e}")
                        
                        time.sleep(delay)
                    
                    print(f"❌ Client {client_id} failed health check after {max_retries} attempts")
                    return False
                
                def start_client_with_retry(client_id, max_retries=5):
                    """Start client with exponential backoff retry logic"""
                    port = Config.client_base_port + client_id
                    url = f'http://{Config.client_address}:{port}/start'
                    
                    for attempt in range(max_retries):
                        try:
                            delay = min(2 ** attempt, 10)  # Exponential backoff with max 10s
                            if attempt > 0:
                                print(f"🔄 Retrying client {client_id} start command (attempt {attempt + 1}/{max_retries})")
                                time.sleep(delay)
                            
                            print(f"🚀 Sending start command to client {client_id} at {url}")
                            response = requests.get(url, timeout=15)
                            
                            if response.status_code == 200:
                                response_data = response.json()
                                print(f"✅ Client {client_id} training started successfully: {response_data}")
                                return True
                            else:
                                print(f"⚠️ Client {client_id} returned status {response.status_code}: {response.text}")
                                
                        except requests.exceptions.RequestException as e:
                            print(f"❌ Network error starting client {client_id}: {e}")
                        except Exception as e:
                            print(f"❌ Unexpected error starting client {client_id}: {e}")
                    
                    print(f"💥 Client {client_id} failed to start after {max_retries} attempts")
                    return False
                
                print("🔍 Performing comprehensive startup synchronization...")
                
                # Phase 1: Wait for all client ports to be available and healthy
                print("📋 Phase 1: Checking client health and readiness...")
                client_health_results = {}
                
                for client_id in range(total_clients):
                    print(f"🔄 Checking health of client {client_id}...")
                    client_health_results[client_id] = check_client_health(client_id)
                
                # Verify all clients are healthy
                failed_clients = [cid for cid, healthy in client_health_results.items() if not healthy]
                if failed_clients:
                    print(f"💥 CRITICAL: Clients {failed_clients} failed health checks. Cannot proceed with training.")
                    return False
                
                print("✅ All clients passed health checks!")
                
                # Phase 2: Send start commands to all clients with retry logic
                print("📋 Phase 2: Initiating training on all clients...")
                start_results = {}
                
                # Use threading for parallel starts but collect results
                def threaded_start(client_id, results_dict):
                    results_dict[client_id] = start_client_with_retry(client_id)
                
                threads = []
                for client_id in range(total_clients):
                    thread = threading.Thread(target=threaded_start, args=(client_id, start_results))
                    thread.daemon = True
                    threads.append(thread)
                    thread.start()
                    time.sleep(0.5)  # Small stagger to avoid overwhelming the system
                
                # Wait for all threads to complete
                for thread in threads:
                    thread.join(timeout=60)  # 60 second timeout per thread
                
                # Phase 3: Verify all clients started successfully
                print("📋 Phase 3: Verifying training initiation results...")
                failed_starts = [cid for cid, success in start_results.items() if not success]
                
                if failed_starts:
                    print(f"💥 CRITICAL: Clients {failed_starts} failed to start training. Training cannot proceed.")
                    print("🔧 Consider checking client logs and restarting the training process.")
                    return False
                
                if len(start_results) != total_clients:
                    missing_clients = [cid for cid in range(total_clients) if cid not in start_results]
                    print(f"💥 CRITICAL: Missing start results for clients {missing_clients}")
                    return False
                
                print("🎉 SUCCESS: All clients have successfully started training!")
                print(f"✅ Training initiated on {total_clients} clients with robust synchronization")
                return True
            
            # Start the training initiation in a separate thread
            training_thread = threading.Thread(target=initiate_training)
            training_thread.daemon = True
            training_thread.start()
            
        except Exception as e:
            # Clean up any started processes on error
            for proc_info in fedshare_processes.values():
                try:
                    proc_info['process'].terminate()
                    proc_info['log_file'].close()
                except:
                    pass
            raise e
    
    def show_logs(self, algorithm):
        """Enhanced log viewer with better formatting"""
        if algorithm not in ['fedshare', 'fedavg', 'scotch']:
            self.send_error(404, "Invalid algorithm")
            return
        
        # Import and reload config to get current values
        import importlib
        import config
        importlib.reload(config)
        
        # Generate dynamic log directory names based on current config
        total_clients = config.Config.number_of_clients
        num_servers = config.Config.num_servers
        
        if algorithm == 'fedavg':
            log_dir_name = f"fedavg-mnist-client-{total_clients}"
        else:
            log_dir_name = f"{algorithm}-mnist-client-{total_clients}-server-{num_servers}"
        
        log_dir = f"logs/{log_dir_name}"
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{algorithm.upper()} Training Logs</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ background: white; padding: 30px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        .back-btn {{ background: linear-gradient(145deg, #95a5a6, #7f8c8d); color: white; padding: 12px 24px; 
                     border: none; border-radius: 8px; text-decoration: none; display: inline-block; margin-bottom: 20px; }}
        .log-file {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 10px; overflow: hidden; }}
        .log-header {{ background: linear-gradient(145deg, #34495e, #2c3e50); color: white; padding: 15px; font-weight: bold; }}
        .log-content {{ background-color: #2c3e50; color: #ecf0f1; padding: 20px; 
                        font-family: 'Courier New', monospace; font-size: 13px; max-height: 400px; 
                        overflow-y: auto; white-space: pre-wrap; line-height: 1.4; }}
        .refresh-btn {{ float: right; background: linear-gradient(145deg, #3498db, #2980b9); color: white; 
                       padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; }}
    </style>
    <script>
        function refreshLogs() {{ location.reload(); }}
        setInterval(refreshLogs, 15000); // Auto-refresh every 15 seconds (less aggressive)
    </script>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Main</a>
        <button class="refresh-btn" onclick="refreshLogs()">🔄 Refresh</button>
        <h1>📋 {algorithm.upper()} Training Logs</h1>"""
        
        if os.path.exists(log_dir):
            log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
            log_files.sort()
            
            for filename in log_files:
                filepath = os.path.join(log_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        if content.strip():  # Only show non-empty logs
                            # Highlight important information
                            content = content.replace('Round:', '<strong>Round:</strong>')
                            content = content.replace('accuracy:', '<span style="color: #2ecc71;"><strong>accuracy:</strong></span>')
                            content = content.replace('loss:', '<span style="color: #e74c3c;"><strong>loss:</strong></span>')
                            content = content.replace('completed', '<span style="color: #f39c12;"><strong>completed</strong></span>')
                            
                    html += f"""
        <div class="log-file">
            <div class="log-header">📄 {filename}</div>
            <div class="log-content">{content}</div>
        </div>"""
                except Exception as e:
                    html += f"<p style='color: red;'>Error reading {filename}: {str(e)}</p>"
        else:
            html += f"""<div style="text-align: center; color: #666; padding: 40px; font-style: italic;">
                No logs found for {algorithm.upper()}.<br>
                <strong>Run the algorithm first to generate training logs.</strong>
            </div>"""
        
        html += """
    </div>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def get_status(self, algorithm):
        if algorithm in running_processes:
            process = running_processes[algorithm]
            if process.poll() is None:
                status = {'status': 'running', 'pid': process.pid}
            else:
                status = {'status': 'completed', 'returncode': process.returncode}
        else:
            status = {'status': 'not_started'}
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())
    
    def get_current_config(self):
        """Get current configuration from config.py"""
        try:
            # Import config to get current values
            import importlib
            import config
            importlib.reload(config)  # Reload to get latest values
            
            current_config = {
                'number_of_clients': config.Config.number_of_clients,
                'num_servers': config.Config.num_servers,
                'training_rounds': config.Config.training_rounds,
                'batch_size': config.Config.batch_size,
                'train_dataset_size': config.Config.train_dataset_size,
                'epochs': config.Config.epochs
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(json.dumps(current_config).encode())
            
        except Exception as e:
            print(f"Error getting current config: {str(e)}")
            self.send_error(500, str(e))
    
    def update_config(self):
        """Update configuration in config.py"""
        try:
            # Get the request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            new_config = json.loads(post_data.decode('utf-8'))
            
            # Validate the configuration
            required_fields = ['clients', 'rounds', 'batch_size', 'train_dataset_size', 'epochs']
            for field in required_fields:
                if field not in new_config:
                    self.send_error(400, f"Missing required field: {field}")
                    return
                if not isinstance(new_config[field], int) or new_config[field] <= 0:
                    self.send_error(400, f"Invalid value for {field}: must be a positive integer")
                    return
            
            # Additional validation for reasonable ranges
            if new_config['clients'] > 20:
                self.send_error(400, "Number of clients cannot exceed 20")
                return
            if new_config['rounds'] > 50:
                self.send_error(400, "Number of rounds cannot exceed 50")
                return
            if new_config['batch_size'] > 1024:
                self.send_error(400, "Batch size cannot exceed 1024")
                return
            if new_config['train_dataset_size'] > 100000:
                self.send_error(400, "Dataset size cannot exceed 100,000")
                return
            if new_config['epochs'] > 20:
                self.send_error(400, "Epochs cannot exceed 20")
                return
            
            # Read current config.py
            with open('config.py', 'r') as f:
                config_content = f.read()
            
            # Update the configuration values
            config_content = re.sub(
                r'number_of_clients = \d+',
                f'number_of_clients = {new_config["clients"]}',
                config_content
            )
            config_content = re.sub(
                r'num_servers = \d+',
                f'num_servers = {new_config["servers"]}',
                config_content
            )
            config_content = re.sub(
                r'train_dataset_size = \d+',
                f'train_dataset_size = {new_config["train_dataset_size"]}',
                config_content
            )
            config_content = re.sub(
                r'training_rounds = \d+',
                f'training_rounds = {new_config["rounds"]}',
                config_content
            )
            config_content = re.sub(
                r'epochs = \d+',
                f'epochs = {new_config["epochs"]}',
                config_content
            )
            config_content = re.sub(
                r'batch_size = \d+',
                f'batch_size = {new_config["batch_size"]}',
                config_content
            )
            
            # Write the updated config back
            with open('config.py', 'w') as f:
                f.write(config_content)
            
            print(f"Configuration updated: {new_config}")
            
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write("Configuration updated successfully!".encode())
            
        except Exception as e:
            print(f"Error updating config: {str(e)}")
            self.send_error(500, str(e))

    def reinitialize_all(self):
        """Kill all clients and servers and reinitialize everything"""
        try:
            global running_processes, progress_data
            
            print("Starting reinitialization: killing all federated learning processes...")
            
            # Kill all federated learning processes by name
            process_names = [
                'fedshareclient.py', 'fedshareserver.py', 'fedshareleadserver.py',
                'fedavgclient.py', 'fedavgserver.py',
                'scotchclient.py', 'scotchserver.py',
                'logger_server.py', 'flask_starter.py'
            ]
            
            for process_name in process_names:
                subprocess.run(['pkill', '-f', process_name], capture_output=True)
            
            # Also kill by algorithm names for broader cleanup
            algorithms = ['fedshare', 'fedavg', 'scotch']
            for algorithm in algorithms:
                subprocess.run(['pkill', '-f', algorithm], capture_output=True)
            
            # Clean up tracked processes
            for algorithm, process in running_processes.items():
                if process and process.poll() is None:
                    try:
                        process.terminate()
                    except:
                        pass
            
            running_processes.clear()
            progress_data.clear()
            
            # Clean up all log directories - use current config to generate names
            import importlib
            import config
            importlib.reload(config)
            
            total_clients = config.Config.number_of_clients
            num_servers = config.Config.num_servers
            
            log_dirs = [
                f'logs/fedshare-mnist-client-{total_clients}-server-{num_servers}',
                f'logs/fedavg-mnist-client-{total_clients}',
                f'logs/scotch-mnist-client-{total_clients}-server-{num_servers}'
            ]
            
            for log_dir in log_dirs:
                subprocess.run(['rm', '-rf', log_dir], capture_output=True)
            
            # Wait a moment for processes to clean up
            time.sleep(2)
            
            print("Reinitialization completed successfully!")
            
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write("All processes killed and system reinitialized successfully!".encode())
            
        except Exception as e:
            print(f"Error during reinitialization: {str(e)}")
            self.send_error(500, f"Reinitialization failed: {str(e)}")

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

def start_server():
    import socketserver
    
    PORT = int(os.getenv('PORT', 5000))
    
    # Create a threaded HTTP server with proper error handling
    class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True
    
    try:
        httpd = ThreadingHTTPServer(("0.0.0.0", PORT), EnhancedFedShareHandler)
        print(f"🚀 Enhanced FedShare server running on http://0.0.0.0:{PORT}", flush=True)
        print("Enhanced interface with real-time progress tracking!", flush=True)
        httpd.serve_forever()
    except OSError as e:
        print(f"Startup error: {e}", flush=True)
        raise

if __name__ == "__main__":
    start_server()