import sys

sys.path.append("./")

from pointnet_autoencoder.dataset.ModelNet import download

def main():
    download("data")

if __name__ == "__main__":
    main()