class Config:
    number_of_clients = 5  # Number of federated learning clients - optimized for reliability
    train_dataset_size = 6000  # Reduced dataset size for faster training
    clients_dataset_size = [train_dataset_size/number_of_clients] * number_of_clients
    total_dataset_size = sum(clients_dataset_size)
    num_servers = 3  # Number of servers (can be modified as needed)
    training_rounds = 3  # Multiple rounds for proper convergence
    epochs = 1
    batch_size = 16  # Larger batch size for faster training
    verbose = 1
    validation_split = 0.1
    server_base_port = 8500
    master_server_index = 0
    master_server_port = 7501
    client_address = '127.0.0.1'
    server_address = '127.0.0.1'
    master_server_address = '127.0.0.1'
    buffer_size = 4096
    client_base_port = 9500
    fedavg_server_port = 3500
    logger_address = '127.0.0.1'
    logger_port = 8778
    delay = 10


class ClientConfig(Config):
    def __init__(self, client_index):
        self.client_index = client_index


class ServerConfig(Config):
    def __init__(self, server_index):
        self.server_index = server_index


class LeadConfig(Config):
    def __init__(self):
        pass


class FedAvgServerConfig(Config):
    def __init__(self):
        pass


# Hierarchical Federated Learning Configuration
class HierConfig(Config):
    # Healthcare Facilities (Clients)
    number_of_facilities = 4  # Healthcare facilities (equivalent to clients)
    facility_base_port = 9600
    
    # Fog Nodes Configuration  
    num_fog_nodes = 3  # Number of fog nodes for intermediate aggregation
    fog_node_base_port = 8600
    
    # Validator Committee Configuration
    committee_size = 3  # Number of validator committee members
    committee_base_port = 8700
    consensus_threshold = 2  # Minimum votes needed (majority)
    
    # Trusted Authority Configuration
    ta_port = 7600
    ta_address = '127.0.0.1'
    
    # Leader Server (randomly selected from fog nodes)
    leader_port = 7650
    
    # Differential Privacy Parameters
    dp_enabled = True  # Enable/disable differential privacy
    dp_mechanism = 'gaussian'  # Differential privacy mechanism
    dp_clip_norm = 1.0  # Gradient clipping norm
    dp_epsilon = 1.0  # Privacy budget (ε)
    dp_delta = 1e-5   # Privacy budget (δ)
    dp_noise_multiplier = 0.1  # Noise multiplier for DP (σ)
    
    # Secret Sharing Parameters (Shamir's)
    secret_sharing_enabled = True  # Enable/disable secret sharing
    secret_num_shares = None  # Number of shares (defaults to num_fog_nodes)
    secret_threshold = 2  # Minimum shares needed to reconstruct
    share_signing_enabled = True  # Enable cryptographic signatures on shares
    
    # Proof-of-Work Parameters for Sybil Resistance  
    pow_difficulty = 4  # Number of leading zeros required in hash
    pow_target = 2**(256 - pow_difficulty)  # Difficulty target
    
    # Byzantine Fault Tolerance
    max_byzantine_nodes = 1  # Maximum number of Byzantine nodes tolerated
    
    # Communication and Security
    enable_encryption = True
    signature_verification = True
    
    # Training Parameters (optimized for hierarchical setup)
    hier_training_rounds = 3  # Updated to match main config
    hier_epochs = 1
    hier_batch_size = 32
    
    # Dataset distribution per facility
    @property
    def facilities_dataset_size(self):
        return [self.train_dataset_size/self.number_of_facilities] * self.number_of_facilities
    
    # Dynamic secret sharing configuration
    @property
    def secret_num_shares_computed(self):
        return self.secret_num_shares if self.secret_num_shares is not None else self.num_fog_nodes


class HierFacilityConfig(HierConfig):
    def __init__(self, facility_index):
        super().__init__()
        self.facility_index = facility_index
        self.facility_port = self.facility_base_port + facility_index


class HierFogNodeConfig(HierConfig):
    def __init__(self, fog_node_index):
        super().__init__()
        self.fog_node_index = fog_node_index
        self.fog_node_port = self.fog_node_base_port + fog_node_index


class HierValidatorConfig(HierConfig):
    def __init__(self, validator_index):
        super().__init__()
        self.validator_index = validator_index
        self.validator_port = self.committee_base_port + validator_index


class HierTrustedAuthorityConfig(HierConfig):
    def __init__(self):
        super().__init__()


class HierLeaderConfig(HierConfig):
    def __init__(self, leader_fog_index=0):
        super().__init__()
        self.leader_fog_index = leader_fog_index  # Which fog node is the leader
