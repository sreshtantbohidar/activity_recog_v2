$p = Get-Process com.docker.backend -ErrorAction SilentlyContinue
if ($p) {
  $c1 = $p.CPU
  Start-Sleep -Seconds 20
  $p.Refresh()
  $d = $p.CPU - $c1
  Write-Output ("backend-cpu-delta-20s: " + [math]::Round($d, 2))
} else {
  Write-Output "NO-BACKEND-PROCESS"
}
Write-Output ("free-c-gb: " + [math]::Round((Get-PSDrive C).Free / 1GB, 1))
