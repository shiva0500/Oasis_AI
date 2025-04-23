; Inno Setup Script for Oasis AI

[Setup]
AppName=Oasis AI: Offline Research Companion
AppVersion=1.0
DefaultDirName={autopf}\OasisAI
DefaultGroupName=Oasis AI
OutputBaseFilename=OasisAIInstaller
Compression=lzma
SolidCompression=yes
OutputDir=output
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "dist\OasisAI.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Oasis AI"; Filename: "{app}\OasisAI.exe"
Name: "{commondesktop}\Oasis AI"; Filename: "{app}\OasisAI.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"
