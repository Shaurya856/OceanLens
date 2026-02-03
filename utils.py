import os
import uuid

def generate_image_id() -> str:
    return str(uuid.uuid4())

def build_enhanced_filename(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    return f"{name}_enhanced{ext}"
