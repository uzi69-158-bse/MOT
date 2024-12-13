import time
from main import main

def benchmark():
    start_time = time.time()
    main()  # Run the main MOT system
    end_time = time.time()
    print(f"Processing Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    benchmark()
