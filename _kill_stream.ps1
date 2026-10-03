$procs = Get-CimInstance Win32_Process -Filter "Name='ssh.exe' OR Name='docker.exe'"
foreach ($p in $procs) {
  $cl = $p.CommandLine
  Write-Output ("PROC " + $p.ProcessId + " :: " + $cl)
  if ($cl -match '192\.168\.1\.125' -or $cl -match 'save.*activity-recognition') {
    Stop-Process -Id $p.ProcessId -Force
    Write-Output ("KILLED " + $p.ProcessId)
  }
}
Write-Output "DONE"
