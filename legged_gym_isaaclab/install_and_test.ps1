# Legged Gym Isaac Lab 扩展 - 安装和测试脚本

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Legged Gym → Isaac Lab Migration" -ForegroundColor Cyan
Write-Host "Installation and Test Script" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 配置
$ISAAC_LAB_PATH = "D:\IsaacLab"
$LEGGED_GYM_EXT_PATH = "D:\legged_gym\legged_gym\legged_gym_isaaclab"
$ISAAC_LAB_EXT_PATH = "$ISAAC_LAB_PATH\source\extensions\legged_gym_isaaclab"

# 步骤 1: 创建软链接
Write-Host "[1/3] Creating symbolic link to Isaac Lab extensions..." -ForegroundColor Yellow

if (Test-Path $ISAAC_LAB_EXT_PATH) {
    Write-Host "   Extension already exists at: $ISAAC_LAB_EXT_PATH" -ForegroundColor Green
    $response = Read-Host "   Do you want to recreate it? (y/N)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Remove-Item $ISAAC_LAB_EXT_PATH -Recurse -Force
        Write-Host "   Removed existing extension" -ForegroundColor Yellow
    } else {
        Write-Host "   Skipping link creation" -ForegroundColor Yellow
        goto TEST
    }
}

try {
    # Try to create symbolic link (requires admin privileges)
    New-Item -ItemType SymbolicLink `
        -Path $ISAAC_LAB_EXT_PATH `
        -Target $LEGGED_GYM_EXT_PATH `
        -ErrorAction Stop | Out-Null
    Write-Host "   ✓ Symbolic link created successfully!" -ForegroundColor Green
} catch {
    # Fallback: copy directory
    Write-Host "   ⚠ Failed to create symbolic link (admin required)" -ForegroundColor Yellow
    Write-Host "   Copying extension instead..." -ForegroundColor Yellow
    Copy-Item -Path $LEGGED_GYM_EXT_PATH -Destination $ISAAC_LAB_EXT_PATH -Recurse -Force
    Write-Host "   ✓ Extension copied successfully!" -ForegroundColor Green
}

:TEST

# 步骤 2: 验证文件
Write-Host "`n[2/3] Verifying installation..." -ForegroundColor Yellow

$required_files = @(
    "extension.toml",
    "setup.py",
    "__init__.py",
    "tasks\__init__.py",
    "tasks\locomotion\__init__.py",
    "tasks\locomotion\legged_robot_env.py",
    "tasks\locomotion\legged_robot_cfg.py",
    "scripts\train.py"
)

$all_ok = $true
foreach ($file in $required_files) {
    $full_path = Join-Path $ISAAC_LAB_EXT_PATH $file
    if (Test-Path $full_path) {
        Write-Host "   ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "   ✗ $file (MISSING!)" -ForegroundColor Red
        $all_ok = $false
    }
}

if (-not $all_ok) {
    Write-Host "`n❌ Installation incomplete! Please check the missing files." -ForegroundColor Red
    exit 1
}

Write-Host "`n   ✓ All files verified!" -ForegroundColor Green

# 步骤 3: 提供测试命令
Write-Host "`n[3/3] Installation complete! Ready to test." -ForegroundColor Yellow

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Option 1: Quick test (512 envs, 100 iterations)" -ForegroundColor Yellow
Write-Host "cd $ISAAC_LAB_PATH" -ForegroundColor White
Write-Host ".\isaaclab.bat -p $LEGGED_GYM_EXT_PATH\scripts\train.py --task Isaac-Legged-Anymal-C-Flat-v0 --num_envs 512 --max_iterations 100 --headless`n" -ForegroundColor Green

Write-Host "Option 2: Full training (2048 envs, 1500 iterations)" -ForegroundColor Yellow
Write-Host "cd $ISAAC_LAB_PATH" -ForegroundColor White
Write-Host ".\isaaclab.bat -p $LEGGED_GYM_EXT_PATH\scripts\train.py --task Isaac-Legged-Anymal-C-Flat-v0 --num_envs 2048 --max_iterations 1500 --headless`n" -ForegroundColor Green

Write-Host "Option 3: Use standard Isaac Lab training script" -ForegroundColor Yellow
Write-Host "cd $ISAAC_LAB_PATH" -ForegroundColor White
Write-Host ".\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Legged-Anymal-C-Flat-v0 --num_envs 2048`n" -ForegroundColor Green

Write-Host "========================================`n" -ForegroundColor Cyan

# 询问是否立即测试
$test_now = Read-Host "Do you want to run a quick test now? (Y/n)"
if ($test_now -ne 'n' -and $test_now -ne 'N') {
    Write-Host "`nStarting quick test..." -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    Set-Location $ISAAC_LAB_PATH
    
    $test_cmd = ".\isaaclab.bat -p `"$LEGGED_GYM_EXT_PATH\scripts\train.py`" --task Isaac-Legged-Anymal-C-Flat-v0 --num_envs 256 --max_iterations 50 --headless"
    
    Write-Host "Running: $test_cmd`n" -ForegroundColor Cyan
    Invoke-Expression $test_cmd
} else {
    Write-Host "`nSetup complete! Run the commands above when ready.`n" -ForegroundColor Green
}
