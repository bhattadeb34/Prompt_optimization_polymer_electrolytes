def load_api_key(file_path: str, keyword: str) -> str:
    """Load API key from file"""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            if keyword.lower() in line.lower():
                key_name, key_value = line.strip().split('=')
                return key_value.strip().strip("'").strip('"')
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None