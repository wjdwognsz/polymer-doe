; =============================================================================
; 🚀 Universal DOE Platform - Inno Setup 설치 스크립트
; =============================================================================
; Windows용 설치 프로그램 생성을 위한 Inno Setup 6.2+ 스크립트
; 다국어 지원, 시스템 요구사항 확인, 자동 업데이트 설정 포함
; =============================================================================

#define MyAppName "Universal DOE Platform"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "DOE Team"
#define MyAppURL "https://universaldoe.com"
#define MyAppExeName "UniversalDOE.exe"
#define MyAppID "{{E3B9C4A2-5F7D-4B8E-9C3A-1D2E5F8A7B9C}"
#define MyAppMutex "UniversalDOEPlatformMutex"

[Setup]
; 앱 기본 정보
AppId={#MyAppID}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/support
AppUpdatesURL={#MyAppURL}/updates
AppCopyright=Copyright (C) 2024 {#MyAppPublisher}

; 설치 경로
DefaultDirName={autopf}\UniversalDOE
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=no
DisableDirPage=no

; 출력 설정
OutputDir=..\dist
OutputBaseFilename=UniversalDOE_Setup_v{#MyAppVersion}
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

; 압축 설정
Compression=lzma2/ultra64
SolidCompression=yes
CompressionThreads=auto

; 권한 설정
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog

; 아키텍처
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

; 버전 정보
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} 설치 프로그램
VersionInfoCopyright=Copyright (C) 2024 {#MyAppPublisher}

; UI 설정
WizardStyle=modern
WizardImageFile=assets\wizard-image.bmp
WizardSmallImageFile=assets\wizard-small-image.bmp
ShowLanguageDialog=yes
LanguageDetectionMethod=uilanguage

; 설치 옵션
AllowNoIcons=yes
AllowUNCPath=no
AllowRootDirectory=no
AllowCancelDuringInstall=yes
CreateUninstallRegKey=yes
UpdateUninstallLogAppName=yes
UsePreviousAppDir=yes
UsePreviousGroup=yes
UsePreviousLanguage=yes
UsePreviousSetupType=yes
UsePreviousTasks=yes
DirExistsWarning=auto

; 디지털 서명 (인증서가 있는 경우)
; SignTool=signtool sign /f "{#SourcePath}\cert.pfx" /p $password /t http://timestamp.digicert.com /d $qUniversal DOE Platform$q $f
; SignedUninstaller=yes

[Languages]
; 다국어 지원
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
; 한국어 메시지
korean.WelcomeLabel1=Universal DOE Platform 설치를 시작합니다
korean.WelcomeLabel2=이 프로그램은 모든 연구자를 위한 AI 기반 실험 설계 플랫폼입니다.%n%n설치를 계속하려면 [다음]을 클릭하세요.
korean.RequiresNet=이 프로그램은 .NET Framework 4.8 이상이 필요합니다.
korean.RequiresVC=Visual C++ 재배포 패키지가 필요합니다.
korean.LaunchProgram=Universal DOE Platform 실행
korean.CreateDesktopIcon=바탕화면 아이콘 생성
korean.CreateQuickLaunchIcon=빠른 실행 아이콘 생성
korean.AssocFileExtension=.doe 파일을 Universal DOE Platform과 연결
korean.OldVersionFound=이전 버전이 설치되어 있습니다. 업그레이드하시겠습니까?
korean.DataFolderInfo=사용자 데이터는 다음 위치에 저장됩니다:
korean.PortInfo=이 프로그램은 포트 8501-8510을 사용합니다.

; 영어 메시지
english.WelcomeLabel1=Welcome to Universal DOE Platform Setup
english.WelcomeLabel2=This will install the AI-powered experiment design platform for all researchers.%n%nClick [Next] to continue.
english.RequiresNet=This application requires .NET Framework 4.8 or later.
english.RequiresVC=Visual C++ Redistributable is required.
english.LaunchProgram=Launch Universal DOE Platform
english.CreateDesktopIcon=Create desktop icon
english.CreateQuickLaunchIcon=Create Quick Launch icon
english.AssocFileExtension=Associate .doe files with Universal DOE Platform
english.OldVersionFound=Previous version detected. Would you like to upgrade?
english.DataFolderInfo=User data will be stored in:
english.PortInfo=This application uses ports 8501-8510.

[Tasks]
; 설치 작업 옵션
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; OnlyBelowVersion: 6.1; Flags: unchecked
Name: "fileassoc"; Description: "{cm:AssocFileExtension}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "firewall"; Description: "Windows 방화벽 예외 추가"; GroupDescription: "시스템 설정"
Name: "startupitem"; Description: "시작 시 자동 실행"; GroupDescription: "시스템 설정"; Flags: unchecked

[Dirs]
; 디렉토리 생성
Name: "{app}"; Permissions: everyone-full
Name: "{app}\data"; Permissions: everyone-full
Name: "{app}\data\logs"; Permissions: everyone-full
Name: "{app}\data\cache"; Permissions: everyone-full
Name: "{app}\data\temp"; Permissions: everyone-full
Name: "{app}\data\db"; Permissions: everyone-full
Name: "{app}\modules"; Permissions: everyone-full
Name: "{app}\modules\user_modules"; Permissions: everyone-full
Name: "{userappdata}\UniversalDOE"; Permissions: everyone-full
Name: "{userappdata}\UniversalDOE\projects"; Permissions: everyone-full
Name: "{userappdata}\UniversalDOE\config"; Permissions: everyone-full

[Files]
; 메인 실행 파일 및 폴더
Source: "..\dist\UniversalDOE\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; 추가 파일
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme

; 환경 설정 템플릿
Source: "assets\.env.example"; DestDir: "{app}"; Flags: ignoreversion

; Visual C++ 재배포 패키지 (포함하는 경우)
; Source: "vcredist\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

; 업데이트 도구
; Source: "updater\updater.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; 시작 메뉴 아이콘
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{group}\사용자 매뉴얼"; Filename: "{app}\docs\manual.pdf"; Flags: createonlyiffileexists
Name: "{group}\Universal DOE 웹사이트"; Filename: "{#MyAppURL}"

; 바탕화면 아이콘
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; IconFilename: "{app}\{#MyAppExeName}"

; 빠른 실행 아이콘
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Registry]
; 프로그램 등록
Root: HKLM; Subkey: "Software\{#MyAppPublisher}"; Flags: uninsdeletekeyifempty
Root: HKLM; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; Flags: uninsdeletekey
Root: HKLM; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"
Root: HKLM; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "Version"; ValueData: "{#MyAppVersion}"
Root: HKLM; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: dword; ValueName: "VersionMajor"; ValueData: "2"
Root: HKLM; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: dword; ValueName: "VersionMinor"; ValueData: "0"

; 파일 연결
Root: HKCR; Subkey: ".doe"; ValueType: string; ValueName: ""; ValueData: "UniversalDOE.Document"; Flags: uninsdeletevalue; Tasks: fileassoc
Root: HKCR; Subkey: "UniversalDOE.Document"; ValueType: string; ValueName: ""; ValueData: "Universal DOE Document"; Flags: uninsdeletekey; Tasks: fileassoc
Root: HKCR; Subkey: "UniversalDOE.Document\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},1"; Tasks: fileassoc
Root: HKCR; Subkey: "UniversalDOE.Document\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: fileassoc

; Windows 앱 목록
Root: HKLM; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppID}"; ValueType: string; ValueName: "DisplayName"; ValueData: "{#MyAppName}"
Root: HKLM; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppID}"; ValueType: string; ValueName: "DisplayVersion"; ValueData: "{#MyAppVersion}"
Root: HKLM; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppID}"; ValueType: string; ValueName: "Publisher"; ValueData: "{#MyAppPublisher}"
Root: HKLM; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppID}"; ValueType: string; ValueName: "URLInfoAbout"; ValueData: "{#MyAppURL}"
Root: HKLM; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppID}"; ValueType: string; ValueName: "DisplayIcon"; ValueData: "{app}\{#MyAppExeName}"
Root: HKLM; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppID}"; ValueType: dword; ValueName: "EstimatedSize"; ValueData: "512000"

; 시작 프로그램 (선택적)
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "{#MyAppName}"; ValueData: """{app}\{#MyAppExeName}"" --minimized"; Flags: uninsdeletevalue; Tasks: startupitem

[Run]
; Visual C++ 재배포 패키지 설치 (필요한 경우)
; Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/quiet /norestart"; StatusMsg: "Visual C++ 재배포 패키지 설치 중..."; Check: VCRedistNeedsInstall

; 방화벽 규칙 추가
Filename: "netsh"; Parameters: "advfirewall firewall add rule name=""{#MyAppName}"" dir=in action=allow program=""{app}\{#MyAppExeName}"" enable=yes"; StatusMsg: "방화벽 예외 추가 중..."; Flags: runhidden; Tasks: firewall
Filename: "netsh"; Parameters: "advfirewall firewall add rule name=""{#MyAppName} Streamlit"" dir=in action=allow protocol=TCP localport=8501-8510 enable=yes"; StatusMsg: "포트 규칙 추가 중..."; Flags: runhidden; Tasks: firewall

; 설치 후 실행
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; 방화벽 규칙 제거
Filename: "netsh"; Parameters: "advfirewall firewall delete rule name=""{#MyAppName}"""; Flags: runhidden
Filename: "netsh"; Parameters: "advfirewall firewall delete rule name=""{#MyAppName} Streamlit"""; Flags: runhidden

[Code]
const
  // 시스템 요구사항
  MIN_WINDOWS_VERSION = '10.0';
  MIN_RAM_MB = 4096;
  MIN_DISK_MB = 2048;
  REQUIRED_DOTNET_VERSION = '4.8';

var
  UpgradeMode: Boolean;
  DataBackupPath: String;

// Windows 버전 확인
function CheckWindowsVersion(): Boolean;
var
  Version: TWindowsVersion;
begin
  GetWindowsVersionEx(Version);
  Result := (Version.Major > 10) or ((Version.Major = 10) and (Version.Minor >= 0));
  
  if not Result then
  begin
    MsgBox('이 프로그램은 Windows 10 이상에서만 실행됩니다.', mbError, MB_OK);
  end;
end;

// RAM 확인
function CheckSystemRAM(): Boolean;
var
  MemoryStatus: TMemoryStatusEx;
begin
  MemoryStatus.dwLength := SizeOf(MemoryStatus);
  if GlobalMemoryStatusEx(MemoryStatus) then
  begin
    Result := (MemoryStatus.ullTotalPhys div (1024 * 1024)) >= MIN_RAM_MB;
    if not Result then
    begin
      MsgBox(Format('이 프로그램은 최소 %d GB의 RAM이 필요합니다.', [MIN_RAM_MB div 1024]), mbError, MB_OK);
    end;
  end
  else
    Result := True; // 확인 실패 시 계속 진행
end;

// 디스크 공간 확인
function CheckDiskSpace(): Boolean;
var
  FreeSpace: Int64;
begin
  FreeSpace := GetSpaceOnDisk(ExpandConstant('{app}'));
  Result := FreeSpace >= (MIN_DISK_MB * 1024 * 1024);
  
  if not Result then
  begin
    MsgBox(Format('설치하려면 최소 %d MB의 여유 공간이 필요합니다.', [MIN_DISK_MB]), mbError, MB_OK);
  end;
end;

// .NET Framework 확인
function IsDotNetInstalled(): Boolean;
var
  NetVersion: String;
begin
  Result := RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full', 
    'Release', NetVersion) and (StrToIntDef(NetVersion, 0) >= 461808); // 4.8
end;

// Visual C++ 재배포 패키지 확인
function VCRedistNeedsInstall(): Boolean;
var
  Installed: Cardinal;
begin
  Result := not RegQueryDWordValue(HKLM, 
    'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) or (Installed <> 1);
end;

// 이전 버전 확인
function GetPreviousVersion(): String;
begin
  if not RegQueryStringValue(HKLM, 'Software\{#MyAppPublisher}\{#MyAppName}', 
    'Version', Result) then
    Result := '';
end;

// 초기화 이벤트
function InitializeSetup(): Boolean;
var
  PrevVersion: String;
  ResultCode: Integer;
begin
  Result := True;
  
  // 시스템 요구사항 확인
  if not CheckWindowsVersion() then
  begin
    Result := False;
    Exit;
  end;
  
  if not CheckSystemRAM() then
  begin
    Result := False;
    Exit;
  end;
  
  // 이전 버전 확인
  PrevVersion := GetPreviousVersion();
  if PrevVersion <> '' then
  begin
    UpgradeMode := True;
    if MsgBox(Format('이전 버전(%s)이 설치되어 있습니다. 업그레이드하시겠습니까?', 
      [PrevVersion]), mbConfirmation, MB_YESNO) = IDNO then
    begin
      Result := False;
      Exit;
    end;
  end;
  
  // 실행 중인 프로세스 확인
  if CheckForMutexes('{#MyAppMutex}') then
  begin
    if MsgBox('Universal DOE Platform이 실행 중입니다. 종료하고 계속하시겠습니까?', 
      mbConfirmation, MB_YESNO) = IDYES then
    begin
      // 프로세스 종료 시도
      Exec('taskkill', '/F /IM {#MyAppExeName}', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
      Sleep(2000); // 2초 대기
      
      if CheckForMutexes('{#MyAppMutex}') then
      begin
        MsgBox('프로그램을 종료할 수 없습니다. 수동으로 종료 후 다시 시도하세요.', mbError, MB_OK);
        Result := False;
        Exit;
      end;
    end
    else
    begin
      Result := False;
      Exit;
    end;
  end;
end;

// 설치 전 준비
procedure CurStepChanged(CurStep: TSetupStep);
var
  DataPath: String;
begin
  case CurStep of
    ssInstall:
    begin
      // 업그레이드 모드에서 데이터 백업
      if UpgradeMode then
      begin
        DataPath := ExpandConstant('{userappdata}\UniversalDOE');
        if DirExists(DataPath) then
        begin
          DataBackupPath := ExpandConstant('{userappdata}\UniversalDOE_Backup_') + 
            FormatDateTime('yyyymmdd_hhnnss', Now());
          RenameFile(DataPath, DataBackupPath);
        end;
      end;
    end;
    
    ssPostInstall:
    begin
      // 데이터 복원
      if (UpgradeMode) and (DataBackupPath <> '') and DirExists(DataBackupPath) then
      begin
        // 백업된 데이터 복원 로직
        // TODO: 구현 필요
      end;
      
      // 환경 설정 파일 생성
      if not FileExists(ExpandConstant('{app}\.env')) then
      begin
        FileCopy(ExpandConstant('{app}\.env.example'), 
          ExpandConstant('{app}\.env'), False);
      end;
    end;
  end;
end;

// 제거 시 데이터 처리
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  case CurUninstallStep of
    usUninstall:
    begin
      if MsgBox('사용자 데이터를 삭제하시겠습니까?'#13#10 + 
        '(프로젝트, 실험 데이터 등)', mbConfirmation, MB_YESNO) = IDNO then
      begin
        // 데이터 유지 - 아무것도 하지 않음
      end
      else
      begin
        // 데이터 삭제
        DelTree(ExpandConstant('{userappdata}\UniversalDOE'), True, True, True);
      end;
    end;
  end;
end;

// 설치 페이지 커스터마이징
procedure InitializeWizard();
var
  InfoPage: TWizardPage;
  InfoMemo: TMemo;
begin
  // 정보 페이지 추가
  InfoPage := CreateCustomPage(wpSelectDir, 
    '중요 정보', 
    '설치 전 확인사항');
    
  InfoMemo := TMemo.Create(InfoPage);
  InfoMemo.Parent := InfoPage.Surface;
  InfoMemo.Left := 0;
  InfoMemo.Top := 0;
  InfoMemo.Width := InfoPage.SurfaceWidth;
  InfoMemo.Height := InfoPage.SurfaceHeight;
  InfoMemo.ScrollBars := ssVertical;
  InfoMemo.ReadOnly := True;
  InfoMemo.Text := 
    '시스템 요구사항:' + #13#10 +
    '- Windows 10 이상 (64비트)' + #13#10 +
    '- 4GB 이상의 RAM' + #13#10 +
    '- 2GB 이상의 여유 공간' + #13#10 +
    '- .NET Framework 4.8' + #13#10 +
    '- Visual C++ 2015-2022 재배포 패키지' + #13#10 + #13#10 +
    '네트워크 설정:' + #13#10 +
    '- 포트 8501-8510 사용' + #13#10 +
    '- 방화벽 예외 필요' + #13#10 + #13#10 +
    '데이터 저장 위치:' + #13#10 +
    ExpandConstant('{userappdata}\UniversalDOE');
end;

// 언어별 라이선스 파일 선택
function GetLicenseFile(): String;
begin
  case ActiveLanguage() of
    'korean': Result := ExpandConstant('{app}\LICENSE.ko.txt');
  else
    Result := ExpandConstant('{app}\LICENSE');
  end;
end;

// 설치 완료 후 추가 작업
procedure DeinitializeSetup();
var
  ResultCode: Integer;
begin
  // 설치 통계 전송 (선택적)
  // Exec(ExpandConstant('{app}\{#MyAppExeName}'), 
  //   '--report-install --version={#MyAppVersion}', '', SW_HIDE, 
  //   ewNoWait, ResultCode);
end;

[Messages]
; 커스텀 메시지
BeveledLabel=Universal DOE Platform v{#MyAppVersion}

[ThirdParty]
; 타사 라이브러리 라이선스 정보
CompileLogMethod=append
