# Headless Training Script for Legged Gym Isaac Lab

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Legged Gym - Headless Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nConfiguration:" -ForegroundColor Yellow
Write-Host "  - Environments: 4"
Write-Host "  - Max Iterations: 100"
Write-Host "  - Mode: Headless (no visualization)"
Write-Host "  - Task: Isaac-Legged-Anymal-C-Flat-v0"

Write-Host "`nStarting training...`n" -ForegroundColor Green

Set-Location D:\IsaacLab

.\isaaclab.bat -p source/extensions/legged_gym_isaaclab/scripts/train.py `
  --task Isaac-Legged-Anymal-C-Flat-v0 `
  --num_envs 4 `
  --max_iterations 100 `
  --headless

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Training Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
