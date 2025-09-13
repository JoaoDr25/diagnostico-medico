
#region conda initialize
# !! Contents within this block are managed by 'conda init' !!
If (Test-Path "C:\Users\Juan Camilo Davila R\miniconda3\Scripts\conda.exe") {
    (& "C:\Users\Juan Camilo Davila R\miniconda3\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | ?{$_} | Invoke-Expression
}
#endregion

