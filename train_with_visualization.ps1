# Training Script with Visualization for Legged Gym Isaac Lab

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Isaac Lab - Legged Gym Training (Visualization Enabled)" -ForegroundColor Cyan  
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  - Environments: 4" -ForegroundColor White
Write-Host "  - Max Iterations: 100" -ForegroundColor White
Write-Host "  - Visualization: Enabled" -ForegroundColor Green
Write-Host "  - Task: Isaac-Legged-Anymal-C-Flat-v0`n" -ForegroundColor White

# Change to IsaacLab directory
Set-Location D:\IsaacLab

# Run training with visualization using Isaac Lab's Python
Write-Host "Starting training...`n" -ForegroundColor Green

.\isaaclab.bat -p source/extensions/legged_gym_isaaclab/scripts/train.py --task Isaac-Legged-Anymal-C-Flat-v0 --num_envs 4 --max_iterations 100

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Training completed!" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

