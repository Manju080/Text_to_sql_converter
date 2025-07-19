# Text-to-SQL Converter - Render.com Deployment

## Quick Deploy to Render.com

### Prerequisites
- Create a free account at [https://render.com](https://render.com)
- Have this folder ready with all files

### Deployment Steps

1. **Push this folder to a Git repository** (GitHub, GitLab, or Bitbucket)
   - Render.com deploys from git repos
   - If you don't have a repo, create one and push:
     ```sh
     git init
     git add .
     git commit -m "Initial Render deployment"
     git remote add origin <your-repo-url>
     git push -u origin main
     ```

2. **Go to [https://dashboard.render.com](https://dashboard.render.com)**

3. **Click 'New +' → 'Web Service'**

4. **Connect your repo and select it**

5. **Configure the service:**
   - **Environment:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port 10000`
   - **(Optional) Add environment variable:** `PYTHON_VERSION=3.9.18`
   - **(Optional) Use `render.yaml` for auto config**

6. **Click 'Create Web Service'**

7. **Wait for build and deployment to finish**

8. **Open your app using the provided URL!**

## API Endpoints
- `/` - Web interface
- `/docs` - API docs
- `/predict` - Predict endpoint
- `/batch` - Batch endpoint
- `/health` - Health check

## Notes
- If your model files are large, Render's free plan may not be enough. Consider upgrading or using external model hosting.
- You can redeploy by pushing new commits to your repo. 