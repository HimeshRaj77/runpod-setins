import os
import subprocess

def deploy_file(local_b64_file, remote_path):
    with open(local_b64_file, "r") as f:
        b64_data = f.read().strip()
    
    cmd = f"echo '{b64_data}' | base64 -d > {remote_path}"
    
    # We use -tt to satisfy RunPod's PTY requirement
    ssh_cmd = [
        "ssh", "-tt", "8birqqgcgxs9ng-6441142c@ssh.runpod.io", cmd
    ]
    
    print(f"Deploying {remote_path}...")
    subprocess.run(ssh_cmd)

if __name__ == "__main__":
    deploy_file("config.b64", "/home/runpod-setins/config.py")
    deploy_file("runpod_start.b64", "/home/runpod-setins/runpod_start.sh")
    deploy_file("llm_engine.b64", "/home/runpod-setins/llm_engine.py")
    
    # Make sure runpod_start.sh is executable
    subprocess.run(["ssh", "-tt", "8birqqgcgxs9ng-6441142c@ssh.runpod.io", "chmod +x /home/runpod-setins/runpod_start.sh"])
    print("Deployment complete!")
