## FAQ, Troubles/疑難排解

### Running Error/執行錯誤

##### [Errno 10013] error while attempting to bind on address ('127.0.0.1', 8090): 嘗試存取通訊端被拒絕，因為存取權限不足

1. Verify Port Availability/確認通訊端是否被佔用

(Open windows command prompt 命令提示字元)

```cmd
netstat -ano | findstr :8090

# If you see output with a PID, note it and check the corresponding process:
tasklist | findstr <PID>

# Kill the process if necessary:
taskkill /PID <PID> /F
```

2. If not occupied, check for Reserved Ports by Windows/若未被佔用，查是否被Windows保留

```cmd
netsh interface ipv4 show excludedportrange protocol=tcp

# If your port (e.g., 8090) is listed, it is reserved by the system.
# To free the port, adjust the dynamic port range:

netsh int ipv4 set dynamicport tcp start=49152 num=16384
netsh int ipv6 set dynamicport tcp start=49152 num=16384

# This command sets the dynamic port range to start from 49152 with a range size of 16384 (default). This excludes ports below 49152, including 8090.

# Restart your computer, then the reserved ports will be set free.
```
