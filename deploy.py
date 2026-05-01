from huggingface_hub import HfApi, whoami
import os

token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN environment variable is not set")

try:
    user_info = whoami(token=token)
    username = user_info['name']
    repo_name = f"{username}/interior-design-ai"
    print(f"Deploying to {repo_name}...")
    
    api = HfApi()
    
    try:
        api.create_repo(repo_id=repo_name, repo_type="space", space_sdk="docker", token=token, exist_ok=True)
        print("Space created or already exists.")
    except Exception as e:
        print(f"Error creating space: {e}")
        
    print("Uploading app.py...")
    api.upload_file(
        path_or_fileobj="app.py",
        path_in_repo="app.py",
        repo_id=repo_name,
        repo_type="space",
        token=token
    )
    
    print("Uploading requirements.txt...")
    api.upload_file(
        path_or_fileobj="requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=repo_name,
        repo_type="space",
        token=token
    )
    
    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="space",
        token=token
    )
    print(f"SUCCESS! View your deployed space at: https://huggingface.co/spaces/{repo_name}")
except Exception as e:
    print(f"Deployment failed: {e}")
