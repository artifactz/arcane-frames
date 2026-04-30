import os, json


def export():
    """Refresh filenames.js"""
    filenames = list_rel_paths("export/images", "export")
    write_filenames(filenames)

def list_rel_paths(folder, rel_to):
    return [os.path.relpath(f"{folder}/{f}", rel_to).replace("\\", "/") for f in os.listdir(folder)]

def write_filenames(filenames: list[str]):
    with open("export/filenames.js", "w") as f:
        f.write("window.FILENAMES = ")
        json.dump(filenames, f)
        f.write(";\n")


if __name__ == "__main__":
    export()
