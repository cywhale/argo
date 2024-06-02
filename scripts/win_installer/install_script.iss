[Setup]
AppName=odbargo_app
AppVersion=0.0.3
DefaultDirName={commonpf}\odbargo_app
DefaultGroupName=odbargo_app
UninstallDisplayIcon={app}\odbargo_app.ico
OutputDir=.
OutputBaseFilename=odbargo_app_installer
DisableDirPage=no

[Files]
Source: "..\..\dist\odbargo_app-0.0.3.tar.gz"; DestDir: "{app}\src"; Flags: ignoreversion
Source: "install_python.bat"; DestDir: "{app}\src"; Flags: ignoreversion
Source: "install_app.bat"; DestDir: "{app}\src"; Flags: ignoreversion
Source: "start_odbargo.bat"; DestDir: "{app}"; Flags: ignoreversion

[Run]
Filename: "{app}\src\install_python.bat"; WorkingDir: "{app}\src"; Flags: shellexec runhidden waituntilterminated; StatusMsg: "Installing Python... Please wait."
Filename: "{app}\src\install_app.bat"; WorkingDir: "{app}\src"; Flags: shellexec runhidden waituntilterminated; StatusMsg: "Installing dependencies... This may take some time. Please wait."

[Icons]
Name: "{group}\odbargo_app"; Filename: "{app}\start_odbargo.bat"; WorkingDir: "{app}"; IconFilename: "{app}\odbargo_app.ico"
Name: "{group}\Uninstall odbargo_app"; Filename: "{uninstallexe}"

[CustomMessages]
SetupMessage=Welcome to the odbargo_app installer. Please ensure that your firewall and any containment software are configured to allow this application to run without restrictions. Failure to do so may result in the application not functioning correctly.

[Code]
procedure InitializeWizard;
begin
  MsgBox(CustomMessage('SetupMessage'), mbInformation, MB_OK);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then begin
    MsgBox('Installation complete. You can now start odbargo_app from the Start menu.', mbInformation, MB_OK);
  end;
end;

[UninstallDelete]
Type: filesandordirs; Name: "{app}\venv"
Type: filesandordirs; Name: "{app}\src"
Type: filesandordirs; Name: "{app}\log"
