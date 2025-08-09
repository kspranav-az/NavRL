#!/usr/bin/env python3
"""
Launch TensorBoard for NavRL logs
"""
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def find_log_directories():
    """Find all possible log directories"""
    possible_dirs = [
        "logs",
        "../logs", 
        "../../logs",
        "./logs/NavRL",
        "../logs/NavRL",
        "../../logs/NavRL"
    ]
    
    existing_dirs = []
    for log_dir in possible_dirs:
        if os.path.exists(log_dir) and os.path.isdir(log_dir):
            # Check if it contains tensorboard files
            for root, dirs, files in os.walk(log_dir):
                if any(f.startswith('events.out.tfevents') for f in files):
                    existing_dirs.append(log_dir)
                    break
    
    return existing_dirs

def launch_tensorboard(log_dir, port=6006):
    """Launch TensorBoard"""
    print(f"🚀 Launching TensorBoard for directory: {log_dir}")
    print(f"📊 TensorBoard will be available at: http://localhost:{port}")
    
    try:
        # Start TensorBoard
        cmd = ["tensorboard", "--logdir", log_dir, "--port", str(port), "--reload_multifile", "true"]
        process = subprocess.Popen(cmd)
        
        # Wait a moment for TensorBoard to start
        time.sleep(3)
        
        # Try to open browser
        try:
            webbrowser.open(f"http://localhost:{port}")
            print("🌐 Browser opened automatically")
        except:
            print("🌐 Please open http://localhost:{port} in your browser")
        
        print("\n💡 Press Ctrl+C to stop TensorBoard")
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping TensorBoard...")
        process.terminate()
        process.wait()
    except Exception as e:
        print(f"❌ Error launching TensorBoard: {e}")
        print("💡 Make sure TensorBoard is installed: pip install tensorboard")

def main():
    print("🔍 Searching for TensorBoard log files...")
    
    log_dirs = find_log_directories()
    
    if not log_dirs:
        print("❌ No TensorBoard log files found!")
        print("💡 Make sure you've run training first to generate logs")
        print("💡 Expected locations:")
        for loc in ["logs/", "../logs/", "./logs/NavRL/"]:
            print(f"   - {loc}")
        return
    
    if len(log_dirs) == 1:
        log_dir = log_dirs[0]
        print(f"✅ Found logs in: {log_dir}")
    else:
        print("✅ Found multiple log directories:")
        for i, log_dir in enumerate(log_dirs):
            print(f"   {i+1}. {log_dir}")
        
        while True:
            try:
                choice = input("Select directory (1-{}): ".format(len(log_dirs)))
                idx = int(choice) - 1
                if 0 <= idx < len(log_dirs):
                    log_dir = log_dirs[idx]
                    break
                else:
                    print("Invalid choice, please try again")
            except ValueError:
                print("Please enter a number")
    
    # Check for port conflicts
    port = 6006
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number, using default 6006")
    
    launch_tensorboard(log_dir, port)

if __name__ == "__main__":
    main()

