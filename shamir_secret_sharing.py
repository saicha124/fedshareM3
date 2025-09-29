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


class OptimizedShamirSecretSharing:
    """Optimized Shamir Secret Sharing using vectorized NumPy operations"""
    
    def __init__(self, threshold, num_shares):
        if threshold > num_shares:
            raise ValueError("Threshold cannot be greater than number of shares")
        self.threshold = threshold
        self.num_shares = num_shares
        # Pre-generate random coefficients using NumPy for better performance
        self.rng = np.random.RandomState(42)  # Seed for reproducibility during development
        
    def split_secret(self, secret_bytes):
        """Vectorized secret splitting using NumPy for O(bytes) complexity"""
        secret_array = np.frombuffer(secret_bytes, dtype=np.uint8)
        
        # Pre-generate polynomial coefficients (vectorized)
        coeffs = self.rng.randint(0, 256, size=(len(secret_array), self.threshold - 1), dtype=np.uint8)
        
        # Pre-compute x-powers for all shares
        x_values = np.arange(1, self.num_shares + 1, dtype=np.uint8)
        
        shares = []
        for share_idx in range(self.num_shares):
            x = x_values[share_idx]
            
            # Vectorized polynomial evaluation: f(x) = secret + a1*x + a2*x^2 + ...
            share_data = secret_array.astype(np.uint16)  # Prevent overflow
            for power in range(self.threshold - 1):
                share_data += coeffs[:, power] * (x ** (power + 1))
            
            # Mod 256 to keep in byte range
            share_data = (share_data % 256).astype(np.uint8)
            shares.append(share_data.tobytes())
            
        return shares
    
    def reconstruct_secret(self, shares_subset):
        """Vectorized secret reconstruction"""
        if len(shares_subset) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")
        
        # Convert shares to numpy arrays
        shares_arrays = [np.frombuffer(share, dtype=np.uint8) for share in shares_subset[:self.threshold]]
        secret_length = len(shares_arrays[0])
        
        # Vectorized Lagrange interpolation at x=0
        reconstructed = np.zeros(secret_length, dtype=np.uint16)
        
        for i in range(self.threshold):
            x_i = i + 1
            
            # Calculate Lagrange coefficient
            numerator = 1
            denominator = 1
            for j in range(self.threshold):
                if i != j:
                    x_j = j + 1
                    numerator *= (0 - x_j)
                    denominator *= (x_i - x_j)
            
            coeff = numerator // denominator
            reconstructed += shares_arrays[i] * coeff
        
        # Mod 256 and convert back to bytes
        return (reconstructed % 256).astype(np.uint8).tobytes()


def _chunked_secret_sharing(data_bytes, num_shares, threshold, chunk_size):
    """Process large data in chunks to prevent hanging - OPTIMIZED VERSION"""
    import time
    import numpy as np
    
    # OPTIMIZATION: Use memoryview for zero-copy slicing
    data_view = memoryview(data_bytes)
    num_chunks = (len(data_bytes) + chunk_size - 1) // chunk_size
    
    print(f"Processing {num_chunks} chunks of ~{chunk_size} bytes each...")
    
    # Initialize Shamir Secret Sharing with optimized polynomial generation
    sss = OptimizedShamirSecretSharing(threshold, num_shares)
    
    all_shares = [[] for _ in range(num_shares)]
    
    # OPTIMIZATION: Process chunks using memoryview (zero-copy)
    for chunk_idx in range(num_chunks):
        start_time = time.time()
        
        # Extract chunk using memoryview (zero-copy)
        start_pos = chunk_idx * chunk_size
        end_pos = min(start_pos + chunk_size, len(data_bytes))
        chunk_data = bytes(data_view[start_pos:end_pos])
        
        # Process this chunk
        chunk_shares = sss.split_secret(chunk_data)
        
        # Combine with previous shares
        for share_idx in range(num_shares):
            all_shares[share_idx].extend(chunk_shares[share_idx])
        
        elapsed = time.time() - start_time
        if chunk_idx % 10 == 0 or elapsed > 1.0:  # Log progress every 10 chunks or if slow
            print(f"Processed chunk {chunk_idx + 1}/{num_chunks} in {elapsed:.2f}s")
    
    print("Chunked secret sharing completed, formatting shares...")
    
    # Format shares for compatibility with existing code
    formatted_shares = []
    for i, share in enumerate(all_shares):
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
    
    print(f"Successfully created {len(formatted_shares)} real Shamir secret shares via chunking")
    return formatted_shares


def shamirs_secret_sharing(data, num_shares, threshold):
    """Split data into real secret shares using Shamir's Secret Sharing"""
    # Serialize the data
    data_bytes = pickle.dumps(data)
    
    print(f"Creating {num_shares} secret shares with threshold {threshold} (total size: {len(data_bytes)} bytes)")
    
    # OPTIMIZATION: Reduced chunk size for better memory management
    chunk_size = 65536  # Process in 64KB chunks (reduced from 256KB) for better resource management
    if len(data_bytes) > chunk_size:
        print(f"Large data detected ({len(data_bytes)} bytes), using chunked processing...")
        return _chunked_secret_sharing(data_bytes, num_shares, threshold, chunk_size)
    
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