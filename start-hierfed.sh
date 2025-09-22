#!/bin/bash

export PYTHONUNBUFFERED=1

# Set memory optimization variables for TensorFlow
export OMP_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

# Read hierarchical configuration from config.py
FACILITIES=$(python -c "from config import HierConfig; print(HierConfig().number_of_facilities)")
FOG_NODES=$(python -c "from config import HierConfig; print(HierConfig().num_fog_nodes)")
VALIDATORS=$(python -c "from config import HierConfig; print(HierConfig().committee_size)")

echo "Hierarchical Federated Learning Configuration:"
echo "  Healthcare Facilities: $FACILITIES"
echo "  Fog Nodes: $FOG_NODES"
echo "  Validator Committee: $VALIDATORS"

DEST_DIRECTORY="hierfed-facilities-${FACILITIES}-fog-${FOG_NODES}-validators-${VALIDATORS}"
echo "Logging to: $DEST_DIRECTORY"
mkdir -p logs/${DEST_DIRECTORY}

echo "Starting Hierarchical Federated Learning System..."

# Step 1: Start Trusted Authority
echo "Starting Trusted Authority..."
nohup python hierta.py &>logs/${DEST_DIRECTORY}/hierta.log &
echo "Trusted Authority started"

# Step 2: Start Validator Committee
echo "Starting Validator Committee..."
for ((VALIDATOR = 0; VALIDATOR < VALIDATORS; VALIDATOR++)); do
  echo "  Starting validator ${VALIDATOR}..."
  nohup python hiervalidator.py "${VALIDATOR}" &>logs/${DEST_DIRECTORY}/hiervalidator-${VALIDATOR}.log &
done
echo "All validators started"

# Step 3: Start Fog Nodes
echo "Starting Fog Nodes..."
for ((FOG = 0; FOG < FOG_NODES; FOG++)); do
  echo "  Starting fog node ${FOG}..."
  nohup python hierfognode.py "${FOG}" &>logs/${DEST_DIRECTORY}/hierfognode-${FOG}.log &
done
echo "All fog nodes started"

# Step 4: Start Leader Server
echo "Starting Leader Server..."
nohup python hierleadserver.py &>logs/${DEST_DIRECTORY}/hierleadserver.log &
echo "Leader server started"

# Wait for infrastructure to initialize
echo "Waiting for infrastructure to initialize..."
sleep 15

# Step 5: Start Healthcare Facilities
echo "Starting Healthcare Facilities..."
for ((FACILITY = 0; FACILITY < FACILITIES; FACILITY++)); do
  echo "  Starting healthcare facility ${FACILITY}..."
  nohup python hierfedclient.py "${FACILITY}" &>logs/${DEST_DIRECTORY}/hierfedclient-${FACILITY}.log &
done
echo "All healthcare facilities started"

# Wait for facilities to initialize and register
echo "Waiting for facilities to register with Trusted Authority..."
sleep 10

# Step 6: Initialize facility registration
echo "Registering healthcare facilities with Trusted Authority..."
for ((FACILITY = 0; FACILITY < FACILITIES; FACILITY++)); do
  FACILITY_PORT=$((9600 + FACILITY))
  echo "  Registering facility ${FACILITY} on port ${FACILITY_PORT}..."
  
  # Send registration request to facility
  curl -s -X POST "http://127.0.0.1:${FACILITY_PORT}/register" &>/dev/null &
done

echo "Registration requests sent to all facilities"
sleep 5

# Step 7: Start initial training round
echo "Initializing first training round..."

# Start round on leader server
curl -s -X POST "http://127.0.0.1:7650/start_round" &>/dev/null

# Trigger first round on all facilities
for ((FACILITY = 0; FACILITY < FACILITIES; FACILITY++)); do
  FACILITY_PORT=$((9600 + FACILITY))
  echo "  Starting training on facility ${FACILITY}..."
  
  # Send empty data for initial round
  curl -s -X POST "http://127.0.0.1:${FACILITY_PORT}/start_round" \
       -H "Content-Type: application/octet-stream" \
       --data-binary "" &>/dev/null &
done

echo ""
echo "🚀 Hierarchical Federated Learning System Started Successfully!"
echo ""
echo "System Components:"
echo "  ✅ Trusted Authority:     http://127.0.0.1:7600"
echo "  ✅ Leader Server:         http://127.0.0.1:7650"
echo "  ✅ Fog Nodes:             ports 8600-$((8600 + FOG_NODES - 1))"
echo "  ✅ Validator Committee:   ports 8700-$((8700 + VALIDATORS - 1))"  
echo "  ✅ Healthcare Facilities: ports 9600-$((9600 + FACILITIES - 1))"
echo ""
echo "Log Directory: logs/${DEST_DIRECTORY}/"
echo ""
echo "Algorithm Features:"
echo "  🔒 Differential Privacy with Gaussian noise"
echo "  🔐 Shamir's Secret Sharing for model parameters"
echo "  🛡️  Byzantine Fault Tolerance via validator committee"
echo "  🏥 CP-ABE encryption for access control"
echo "  ⚙️  Proof-of-Work for Sybil resistance"
echo "  🌫️  Hierarchical aggregation via fog nodes"
echo ""
echo "The system will now run autonomously. Check logs for detailed progress."
echo "Training will proceed through multiple rounds with privacy-preserving aggregation."

# Keep script running to monitor (optional)
python flask_starter.py