import sys
import os
from datetime import datetime

# Function to initialize data path
def init_datapath(type="default"):
    # Check if the operating system is Windows
    if os.name == 'nt':  
        # Default dataset path for Windows
        dataset_path = "E://Dataset//selfmade_Coloradar//"
        
        # If the default path exists, use it; otherwise, use the alternate path
        if not os.path.exists(dataset_path):
            dataset_path = "D://SelfColoradar//DataCaptured//"
        
        # If the type is "rotation", use a different path
        if type == "rotation":
            dataset_path = "E://Dataset//selfmade_Coloradar_rotation//"
    else:  # For Unix-like systems (Linux, macOS, etc.)
        # Default dataset path for Unix-like systems
        dataset_path = "/share2/data/ruixu/RadarEyes/"
        
    return dataset_path

# Class to handle dual output (console and file)
class DualOutput:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout

    # Function to write text to both file and stdout
    def write(self, text):
        self.file.write(text)
        self.stdout.write(text)

    # Function to flush both file and stdout
    def flush(self):
        self.file.flush()
        self.stdout.flush()

# Function to get the source code of the current script
def get_source_code():
    with open(sys.argv[0], 'r', encoding='utf-8') as file:
        return file.read()

def redirect_output_to_logfile():
    # Get the filename of the current script (excluding the extension)
    script_path = os.path.abspath(sys.argv[0])
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a 'log' directory if it doesn't exist
    log_directory = './log'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Create log filename using the script's filename and timestamp
    log_filename = f"{script_name}_{timestamp}.log"

    # Change the log file path to inside the 'log' directory
    log_filepath = os.path.join(log_directory, log_filename)

    # Open the log file, redirect output to this file and the console
    log_file = open(log_filepath, 'w', encoding='utf-8')

    # Add separator lines and blank lines to the log file
    separator = "-" * 40 + "\n" +  "-" * 40
    log_file.write(separator + "\n")

    # Add source code to the log file
    log_file.write("Part1: Original Code: " + script_path + "\n")
    log_file.write("\n" + separator + "\n")
    log_file.write(get_source_code())
    log_file.write("\n" + separator + "\n")
    log_file.write("\n" * 3)

    # Add run command to the log file
    log_file.write("\n" + separator + "\n")
    log_file.write("Part2: Run Command: " + os.path.abspath(sys.executable) + " " + " ".join(sys.argv) + "\n")
    log_file.write(separator + "\n")
    log_file.write("\n" * 3)

    # Record output
    log_file.write("\n" + separator + "\n")
    log_file.write("Part3: Output: " + " ".join(sys.argv) + "\n")
    dual_output = DualOutput(log_file)
    sys.stdout = dual_output
    sys.stderr = sys.stdout