import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Expected a boolean value.")
    
def prompt_overwrite_confirmation(message, logger=None, force_overwrite=False):
    """
    Prompt the user for confirmation before overwriting files.
    Returns True if user confirms or if force_overwrite is True, False otherwise.
    """
    if force_overwrite:
        if logger:
            logger.log("Force overwrite enabled: skipping user prompt.")
        return True
    
    print("\n" + "=" * 80)
    print("OVERWRITE WARNING")
    print("=" * 80)
    print(message)
    print("=" * 80)
    
    while True:
        try:
            response = input("Do you want to proceed and overwrite existing files? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                if logger:
                    logger.log("User confirmed: proceeding with overwrite.")
                return True
            elif response in ['n', 'no']:
                if logger:
                    logger.log("User declined: exiting without overwrite.")
                print("Exiting without overwrite.")
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        except (EOFError, KeyboardInterrupt):
            print("\nOperation cancelled by user.")
            if logger:
                logger.log("Operation cancelled by user.")
            return False
