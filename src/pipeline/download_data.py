import os
import requests

def download_file(url, save_path):
    print(f"Downloading {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Saved to {save_path}")
    else:
        print(f"Failed to download {url} (Status: {response.status_code})")

def main():
    base_url = "https://raw.githubusercontent.com/martj42/international_results/master"
    
    files = {
        "results.csv": f"{base_url}/results.csv",
        "shootouts.csv": f"{base_url}/shootouts.csv"
    }
    
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/match"))
    
    for filename, url in files.items():
        save_path = os.path.join(data_dir, filename)
        download_file(url, save_path)

if __name__ == "__main__":
    main()
