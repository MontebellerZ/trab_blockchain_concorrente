# thread_miner.py

import hashlib
import time
import threading
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np

# Classe para gerenciar os resultados da mineração
class MiningResults:
    def __init__(self):
        self.results = []
        self.filename = "mining_results.csv"
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'block_index', 'threads', 'elapsed_time', 'nonce'])

    def add_result(self, block_index, threads, elapsed_time, nonce):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.results.append({
            'timestamp': timestamp,
            'block_index': block_index,
            'threads': threads,
            'elapsed_time': elapsed_time,
            'nonce': nonce
        })
        
        # Salva no CSV
        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, block_index, threads, elapsed_time, nonce])
        
        # Atualiza os gráficos
        self.update_dots_plot()
        self.update_bar_plot()

    def update_bar_plot(self):
        # Lê os dados do CSV
        df = pd.read_csv(self.filename)
        
        # Agrupa os dados por número de threads
        grouped = df.groupby('threads')['elapsed_time']
        thread_counts = sorted(df['threads'].unique())
        avg_times = [grouped.get_group(t).mean() for t in thread_counts]
        
        # Configuração do gráfico de barras
        plt.figure(figsize=(5, 5))
        bar_width = 0.5
        index = np.arange(len(thread_counts))
        
        # Plota as barras para média, mínimo e máximo
        plt.bar(index, avg_times, bar_width, label='Média', color='#3498db')
        
        # Configurações do gráfico
        plt.title('Média Tempo de Mineração por Nº Threads')
        plt.xlabel('Número de Threads')
        plt.ylabel('Tempo (segundos)')
        plt.xticks(index + bar_width, thread_counts)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--')
        plt.tight_layout()
        
        plt.savefig('threads_vs_time_bar.png')

    def update_dots_plot(self):
        threads = []
        times = []
        with open(self.filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                threads.append(int(row['threads']))
                times.append(float(row['elapsed_time']))
        
        plt.figure(figsize=(10, 6))
        plt.scatter(threads, times, alpha=0.5)
        plt.title('Threads vs Tempo de Mineração')
        plt.xlabel('Número de Threads')
        plt.ylabel('Tempo (segundos)')
        plt.grid(True)
        plt.savefig('threads_vs_time_dots.png')

# Classe que representa um bloco da blockchain
class Block:
    def __init__(self, index: int, previous_hash: str, timestamp: float, data, nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.nonce = nonce
        self.hash = self.calculate_hash()

    # Funcao que calcula o hash do bloco
    def calculate_hash(self):
        value = str(self.index) + self.previous_hash + str(self.timestamp) + self.data + str(self.nonce)
        return hashlib.sha256(value.encode()).hexdigest()

    # Funcao que executa a prova de trabalho (mineracao)
    def mine_block(self, difficulty, stop_event):
        prefix = '0' * difficulty

        while not self.hash.startswith(prefix):
            if stop_event.is_set():
                return
            self.nonce += 1
            self.hash = self.calculate_hash()

        if stop_event.is_set():
            return
        
        # Avisar que um bloco foi minerado
        stop_event.set()

# Blockchain basica com lista de blocos
class Blockchain:
    def __init__(self, difficulty=4):
        self.chain = [self.create_genesis_block()]
        self.difficulty = difficulty

    def create_genesis_block(self):
        return Block(0, "0", time.time(), "Genesis Block")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block: Block):
        last_block = self.get_latest_block()
        new_block.index = last_block.index + 1
        new_block.previous_hash = last_block.hash
        self.chain.append(new_block)

    def get_latest_block_index(self):
        return len(self.chain)

# Funcao para mineracao concorrente com multiplas threads
def concurrent_mining(num_threads: int, difficulty: int, num_blocks: int, results_manager: MiningResults):
    print(f"\nMineracao com {num_threads} threads iniciada.")

    using_blockchain = Blockchain(difficulty)

    while using_blockchain.get_latest_block_index() <= num_blocks:
        latest_block = using_blockchain.get_latest_block()
        stop_event = threading.Event()

        block_copy = Block(latest_block.index, latest_block.previous_hash, latest_block.timestamp, latest_block.data)

        def mine():
            block_copy.mine_block(difficulty, stop_event)

        threads: list[threading.Thread] = []

        start_time = time.time()
        for _ in range(num_threads):
            t = threading.Thread(target=mine)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        new_block = Block(0, "0", time.time(), "novo bloco")
        using_blockchain.add_block(new_block)

        elapsed_time = time.time() - start_time
        print(f"Bloco #{block_copy.index} minerado ({elapsed_time:.2f}s) com nonce {block_copy.nonce}: {block_copy.hash}")

        # Salva os resultados
        results_manager.add_result(block_copy.index, num_threads, elapsed_time, block_copy.nonce)
        
    print(f"Mineracao com {num_threads} threads concluída.")

# Ponto de entrada principal
if __name__ == "__main__":
    num_blocks = 2
    difficulty = 4
    thread_amounts = [1, 2, 4, 8]
    
    results_manager = MiningResults()

    for num_threads in thread_amounts:
        concurrent_mining(num_threads, difficulty, num_blocks, results_manager)