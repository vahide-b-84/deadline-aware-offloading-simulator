# server.py
import simpy

class Server:
    def __init__(self, env, server_type, server_id, processing_frequency):
        self.env = env
        self.server_type = server_type
        self.server_id = server_id

        # Simple FIFO queue (no priorities)
        self.queue = simpy.Resource(env, capacity=1)

        self.processing_frequency = processing_frequency
