import os
import shutil
import subprocess
import sys
import zipapp
from pathlib import Path

def fix_pth_files(target_dir: Path):
    """
    Finds .pth files in target_dir, copies the actual source folders 
    they point to, and removes the .pth file.
    """
    # Search for all .pth files (e.g., __editable___myapp_0_1_0.pth)
    for pth_file in list(target_dir.glob("*.pth")):
        with open(pth_file, "r") as f:
            source_path_str = f.read().strip()
            
        source_path = Path(source_path_str).resolve()
        
        if source_path.exists() and source_path.is_dir():
            print(f"Replacing link {pth_file.name} with physical copy of {source_path}")
            
            for item in source_path.iterdir():
                if item.is_dir() and not item.name.startswith(('.', '__')):
                    dest = target_dir / item.name
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
            
            pth_file.unlink()

def run_command(command: list[str], cwd=None):
    """Utility to run shell commands via uv."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout

def build_zipapp(package_dir: Path, entry_point: str, output_path: Path):
    dist_path = Path("dist")
    build_path = dist_path / "build"
    
    # 1. Clean up previous builds
    if dist_path.exists():
        shutil.rmtree(dist_path)
    dist_path.mkdir()
    build_path.mkdir()

    # 2. Install packages
    print(f"--- Installing to Target (Physical Copy) ---")
    run_command([
        "uv", "pip", "install", ".",
        "--target", str(build_path), 
        "--reinstall",
        "--refresh",
        "--link-mode=copy",
        "--no-cache"
    ], package_dir)

    # 3. Fix the .pth files manually
    print("--- Manually replacing .pth links with actual code ---")
    fix_pth_files(package_dir / build_path)

    # 4. Define Filter
    EXCLUDED_PACKAGES = {"numpy", "pydantic", "pydantic_core", "pillow", "beet", "nbtlib", "PIL"}

    def zip_filter(path: Path):
        # Check if any part of the path is in our excluded list
        for part in path.parts:
            if part in EXCLUDED_PACKAGES:
                return False
        # Also exclude __pycache__ and dist-info to save space
        if "__pycache__" in path.parts or path.suffix == ".dist-info":
            return False
        return True

    # 5. Create the zipapp
    print(f"--- Creating Zipapp ---")
    zipapp.create_archive(
        source=package_dir / build_path,
        target=output_path,
        main=entry_point,
        filter=zip_filter
    )

    print(f"Successfully created: {output_path}")
    shutil.rmtree(build_path)

if __name__ == "__main__":
    build_zipapp(
        package_dir=Path("aegis-server"), 
        entry_point="aegis_server.__main__:main", 
        output_path=Path("aegis-vscode/language_server.pyz")
    )