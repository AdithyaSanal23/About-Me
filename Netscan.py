import socket
from datetime import datetime

# Target input
target = input("Enter the IP address or hostname to scan: ")
start_port = int(input("Enter the starting port: "))
end_port = int(input("Enter the ending port: "))

# Resolve hostname to IP
try:
    target_ip = socket.gethostbyname(target)
except socket.gaierror:
    print("Hostname could not be resolved.")
    exit()

print(f"\nScanning {target_ip} from port {start_port} to {end_port}")
print("Scan started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Port scanning loop
for port in range(start_port, end_port + 1):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)  # Timeout for each connection attempt

    result = s.connect_ex((target_ip, port))
    if result == 0:
        print(f"[+] Port {port} is OPEN")
    s.close()

print("Scan completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
