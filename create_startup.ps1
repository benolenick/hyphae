$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut("$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\Hyphae Server.lnk")
$s.TargetPath = "C:\Users\om\AppData\Local\Programs\Python\Python311\pythonw.exe"
$s.Arguments = "C:\Users\om\Desktop\hyphae\start_server.py"
$s.WorkingDirectory = "C:\Users\om\Desktop\hyphae"
$s.WindowStyle = 7
$s.Save()
Write-Host "Startup shortcut created"
