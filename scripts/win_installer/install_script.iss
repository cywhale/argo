#define AppVersion "0.0.5"

[Setup]
AppName=odbargo_app
AppVersion={#AppVersion}
DefaultDirName={commonappdata}\odbargo_app
DefaultGroupName=odbargo_app
UninstallDisplayIcon={app}\odbargo_app.ico
OutputDir=.
OutputBaseFilename=odbargo_app_installer
DisableDirPage=no
SetupLogging=yes

[Dirs]
Name: "{app}\log"; Permissions: everyone-full

[Files]
Source: "..\..\dist\requirements.txt"; DestDir: "{app}\src"; Flags: ignoreversion
Source: "..\..\dist\odbargo_app.tar.gz"; DestDir: "{app}\src"; Flags: ignoreversion
Source: "python_downloader_bits.ps1"; DestDir: "{app}\src"; Flags: ignoreversion
Source: "install_env.bat"; DestDir: "{app}\src"; Flags: ignoreversion
Source: "check_port.bat"; DestDir: "{app}\src"; Flags: ignoreversion
Source: "start_odbargo.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "uninstall_env.bat"; DestDir: "{app}\src"; Flags: ignoreversion

[Run]
Filename: "{app}\src\install_env.bat"; WorkingDir: "{app}\src"; Flags: shellexec runhidden waituntilterminated; StatusMsg: "Installing python dependencies. This may take some time. Please wait."

[Icons]
Name: "{group}\odbargo_app"; Filename: "{app}\start_odbargo.bat"; WorkingDir: "{app}"; IconFilename: "{app}\odbargo_app.ico"
Name: "{group}\Uninstall odbargo_app"; Filename: "{uninstallexe}"

[CustomMessages]
SetupMessage=Welcome to the odbargo_app installer. Please ensure that your firewall and any containment software are configured to allow this application to run without restrictions. Failure to do so may result in the application not functioning correctly.

[Code]
var
  CustomLabel: TLabel;

procedure InitializeWizard;
begin
  MsgBox(CustomMessage('SetupMessage'), mbInformation, MB_OK);
  
  CustomLabel := TLabel.Create(WizardForm);
  CustomLabel.Parent := WizardForm.InstallingPage;
  CustomLabel.Left := 0;
  CustomLabel.Top := WizardForm.ProgressGauge.Top + WizardForm.ProgressGauge.Height + 8;
  CustomLabel.Width := WizardForm.ClientWidth;
  CustomLabel.Height := 40;
  CustomLabel.Caption := ' ';
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpInstalling  then
  begin
    CustomLabel.Caption := 'Watch the log/install_env.log to see the installation progress.';
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then begin
    MsgBox('Installation complete. You can now start odbargo_app from the Start menu.' + #13#10 + 'Watch the log/install_env.log to see the installation progress.', mbInformation, MB_OK);
    // Inform user about port
    if FileExists(ExpandConstant('{app}\config_app.yaml')) then
    begin
      MsgBox('The application will run on the port specified in config_app.yaml. You can manually alter the port by modifying this file.', mbInformation, MB_OK);
    end;	
  end;
end;

function InitializeUninstall(): Boolean;
var
  InstallationFolder: string;
  ResultCode: Integer;
  ShouldRemoveEnv: Integer;
begin
  Result := MsgBox('Uninstall is initializing. Do you really want to uninstall this App?', mbConfirmation, MB_YESNO) = idYes;
  if not Result then
    Abort();

  InstallationFolder := ExpandConstant('{app}');
  ShouldRemoveEnv := MsgBox('Do you want to remove the Python virtual environment created by this App?', mbConfirmation, MB_YESNO);

  if ShouldRemoveEnv = idYes then
  begin
    if Exec('cmd.exe', '/c "' + InstallationFolder + '\src\uninstall_env.bat" true', '', SW_SHOW, ewWaitUntilTerminated, ResultCode) then
      Log('Succeeded running uninstall_env.bat with parameter true')
    else
      Log('Failed running uninstall_env.bat with parameter true');
  end
  else
  begin
    if Exec('cmd.exe', '/c "' + InstallationFolder + '\src\uninstall_env.bat" false', '', SW_SHOW, ewWaitUntilTerminated, ResultCode) then
      Log('Succeeded running uninstall_env.bat with parameter false')
    else
      Log('Failed running uninstall_env.bat with parameter false');
  end;

  Result := True;  // Ensure the uninstall continues
end;

procedure DeinitializeUninstall();
begin
  Log(Format('DeinitializeUninstall triggered at %s', [GetDateTimeString('yyyy-mm-dd hh:nn:ss', '-', ':')]));
end;

procedure CurUninstallStepChanged(UninstallStep: TUninstallStep);
begin
  Log(Format('CurUninstallStepChanged triggered at %s', [GetDateTimeString('yyyy-mm-dd hh:nn:ss', '-', ':')]));
  Log('CurUninstallStepChanged step: ' + IntToStr(Integer(UninstallStep)));
end;

[UninstallDelete]
Type: filesandordirs; Name: "{app}\log"
Type: filesandordirs; Name: "{app}\src"
Type: filesandordirs; Name: "{app}\venv"
Type: filesandordirs; Name: "{app}\config_app.yaml"