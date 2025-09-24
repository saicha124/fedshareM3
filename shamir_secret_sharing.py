#!/usr/bin/env python3
"""
Real Shamir Secret Sharing Implementation
Using polynomial interpolation over finite field GF(2^8) for security
"""
import random
import numpy as np
import pickle
import base64


class ShamirSecretSharing:
    """Real Shamir Secret Sharing implementation using polynomial interpolation"""
    
    def __init__(self, threshold, num_shares):
        self.threshold = threshold
        self.num_shares = num_shares
        self.prime = 257  # Use prime for finite field operations
        
    def _mod_inverse(self, a, m=None):
        """Calculate modular multiplicative inverse using extended Euclidean algorithm"""
        if m is None:
            m = self.prime
        
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a % m, m)
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        return (x % m + m) % m
    
    def _polynomial_eval(self, coefficients, x):
        """Evaluate polynomial at point x using Horner's method"""
        result = 0
        for coeff in reversed(coefficients):
            result = (result * x + coeff) % self.prime
        return result
    
    def _lagrange_interpolation(self, shares, x=0):
        """Reconstruct secret using Lagrange interpolation"""
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares, got {len(shares)}")
        
        # Use first 'threshold' shares for reconstruction
        shares = shares[:self.threshold]
        
        result = 0
        for i, (xi, yi) in enumerate(shares):
            # Calculate Lagrange basis polynomial
            numerator = 1
            denominator = 1
            
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (x - xj)) % self.prime
                    denominator = (denominator * (xi - xj)) % self.prime
            
            # Calculate modular inverse of denominator
            denominator_inv = self._mod_inverse(denominator)
            lagrange_basis = (numerator * denominator_inv) % self.prime
            
            result = (result + yi * lagrange_basis) % self.prime
        
        return result % self.prime
    
    def split_secret(self, secret_bytes):
        """Split secret bytes into shares using polynomial interpolation"""
        shares = [[] for _ in range(self.num_shares)]
        
        # Process each byte of the secret
        for byte_val in secret_bytes:
            # Generate random coefficients for polynomial of degree (threshold-1)
            coefficients = [byte_val]  # Secret is the constant term
            for _ in range(self.threshold - 1):
                coefficients.append(random.randint(0, self.prime - 1))
            
            # Evaluate polynomial at points 1, 2, ..., num_shares
            for i in range(self.num_shares):
                x = i + 1  # Use 1-based indexing to avoid x=0
                y = self._polynomial_eval(coefficients, x)
                shares[i].append((x, y))
        
        return shares
    
    def reconstruct_secret(self, shares_subset):
        """Reconstruct secret from subset of shares"""
        if len(shares_subset) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")
        
        # Determine secret length from first share
        secret_length = len(shares_subset[0])
        secret_bytes = bytearray()
        
        # Reconstruct each byte
        for byte_idx in range(secret_length):
            # Extract shares for this byte position
            byte_shares = []
            for share in shares_subset:
                if byte_idx < len(share):
                    byte_shares.append(share[byte_idx])
            
            if len(byte_shares) >= self.threshold:
                # Reconstruct this byte using Lagrange interpolation
                reconstructed_byte = self._lagrange_interpolation(byte_shares, x=0)
                secret_bytes.append(reconstructed_byte)
        
        return bytes(secret_bytes)


def shamirs_secret_sharing(data, num_shares, threshold):
    """Split data into real secret shares using Shamir's Secret Sharing"""
    # Serialize the data
    data_bytes = pickle.dumps(data)
    
    print(f"Creating {num_shares} secret shares with threshold {threshold} (total size: {len(data_bytes)} bytes)")
    
    # Initialize Shamir Secret Sharing
    sss = ShamirSecretSharing(threshold, num_shares)
    
    # Split the secret
    raw_shares = sss.split_secret(data_bytes)
    
    # Format shares for compatibility with existing code
    formatted_shares = []
    for i, share in enumerate(raw_shares):
        # Encode share as base64 for JSON compatibility
        share_bytes = pickle.dumps(share)
        share_data = {
            'share_id': i + 1,
            'data_fragment': base64.b64encode(share_bytes).decode('utf-8'),
            'size_info': {
                'index': i,
                'total': num_shares,
                'total_size': len(data_bytes),
                'share_size': len(share_bytes)
            },
            'threshold': threshold,
            'total_shares': num_shares,
            'is_real_sss': True  # Mark as real SSS
        }
        formatted_shares.append(share_data)
    
    print(f"Successfully created {len(formatted_shares)} real Shamir secret shares")
    return formatted_shares


def reconstruct_secret_shares(shares_by_facility):
    """Reconstruct model parameters from real Shamir secret shares"""
    facility_models = {}
    
    for facility_id, facility_shares_dict in shares_by_facility.items():
        print(f"Reconstructing facility {facility_id} from {len(facility_shares_dict)} shares")
        
        try:
            # Extract and decode shares
            raw_shares = []
            threshold = None
            total_shares = None
            
            for share_id, share_info in facility_shares_dict.items():
                share_data = share_info['share']
                
                # Get threshold and total_shares from first share
                if threshold is None:
                    threshold = share_data.get('threshold', 2)
                    total_shares = share_data.get('total_shares', 3)
                
                # Check if this is real SSS
                if not share_data.get('is_real_sss', False):
                    print(f"Warning: Share {share_id} is not real SSS format")
                    continue
                
                # Decode the share
                share_bytes = base64.b64decode(share_data['data_fragment'])
                share = pickle.loads(share_bytes)
                raw_shares.append(share)
            
            if len(raw_shares) < threshold:
                print(f"Insufficient shares for facility {facility_id}: {len(raw_shares)} < {threshold}")
                continue
            
            # Initialize SSS with correct parameters
            sss = ShamirSecretSharing(threshold, total_shares)
            
            # Reconstruct the secret
            reconstructed_bytes = sss.reconstruct_secret(raw_shares)
            
            # Deserialize the model parameters
            model_params = pickle.loads(reconstructed_bytes)
            facility_models[facility_id] = model_params
            
            print(f"Successfully reconstructed facility {facility_id} model using real SSS")
            
        except Exception as e:
            print(f"Error reconstructing facility {facility_id}: {e}")
            continue
    
    print(f"Reconstructed models from {len(facility_models)} facilities using real Shamir Secret Sharing")
    return facility_models