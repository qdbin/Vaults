/*
作者微信：sunwind1576157
最新版本：https://blog.csdn.net/liuyukuan/article/details/120558428
功能说明：
1、为Obsidian增加关闭到托盘功能
2、增加托盘图标，左键点击可以显示/隐藏窗口
3、可以用Win+z热键恢复显示或隐藏窗口
注意，须配置下面的app为您Obsidian本地路径
 
————————————————
版权声明：本文为CSDN博主「liuyukuan」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/liuyukuan/article/details/120558428
 
*/
#Persistent
#SingleInstance force
DetectHiddenWindows, On
SetWorkingDir %A_ScriptDir% 
CoordMode,Mouse,Screen
 
WinShow %theWin%
 
global app
global WinId
;请修改路径为您本地的Obsidian程序路径
IniRead,app,cfg.ini,config,app
 
;或者直接在AHK脚本中直接指定您本地的Obsidian程序路径
;app:="C:\Users\hanbin\scoop\apps\obsidian\current\Obsidian.exe"
 
SplitPath, app, theExe
target:="ahk_exe " theExe
 
Menu,Tray,Icon,%app%
Menu,Tray,NoStandard
 
Menu,Tray,add,显示/隐藏(&O),open
 
Menu,Tray,add,
 
Menu,Tray,add,运行Obsidian(&S),runObsidian
Menu, Tray, Add, 重启脚本(&R), Reload
Menu,Tray,add,
Menu,Tray,add,关于(&A),about
Menu,Tray,add,退出(&E),Exit
Menu,Tray,Tip,Obsidian助手
; Menu,Tray,add,设置(&S),setting
 
Menu, Tray, Default, 显示/隐藏(&O)
Menu, Tray, Click, 1
 
gosub runObsidian
 
SysGet,SM_CXSIZEFRAME,32
SysGet,SM_CYSIZEFRAME,33
SysGet,SM_CXSIZE,30
SysGet,SM_CYSIZE,31
Loop
{
  Sleep,100
  MouseGetPos,mx,my,_WinId
  WinGetPos,x,y,w,h,ahk_id %_WinId%
;关闭按钮
  l:=x+w-SM_CXSIZEFRAME-SM_CXSIZE-10
  t:=y-SM_CYSIZEFRAME
  r:=x+w-SM_CXSIZEFRAME
  b:=y+SM_CYSIZE+SM_CYSIZEFRAME
  If (mx<l Or mx>r Or my<t Or my>b)
  {
    Hotkey,LButton,onClick,Off
    ToolTip
    }
  Else
  {
    WinGet,program,ProcessName,ahk_id %_WinId%  
    if (Program<>theExe)
            Continue
    WinId:=_WinId
    Hotkey,LButton,onClick,On 
    ToolTip,最小化到托盘
  }
}
Return
 
onClick:
theWin:=getWin(target)
;WinMiniMize %theWin%
winhide ahk_id %WinId%
Return
 
 ;全局显示隐藏快捷键 - 直接隐藏到托盘
#o::
    theWin:=getWin(target)
    if DllCall("IsWindowVisible", "Ptr", theWin) 
    {
        WinHide %theWin%  ; 隐藏到托盘
    }
    else
    {
        WinShow %theWin%  ; 显示窗口
        WinActivate %theWin%
    }
    Return
 
open:
ShowOrHide(WinId)
Return
 
 
About:
Gui, 2:Add, Text, ,联系作者：
Gui, 2:Font, underline
Gui, 2:Add, Text, ym cRed vHyperlink_mail gSendMail, 157157@qq.com
Gui, 2:Font, norm
Gui, 2:Add, Text, xm section,访问网站：
Gui, 2:Font, underline
Gui, 2:Add, Text, ys cBlue  vHyperlink_website gLaunchwebsite, blog.csdn.net/liuyukuan
Gui, 2:Font, norm
Gui, 2:Add, Text,xm ,微信打赏：  sunwind1576157
Gui, 2:Add, Button, w50 xp+200 yp-5 Default, OK
hCursor:=DllCall("LoadCursor",UInt,0,UInt,32649)
onMessage(0x200,"WM_MOUSEMOVE")
Gui, 2:Show,,关于
return
 
2ButtonOK:
2GuiClose:
2GuiEscape:
Gui, 2:Destroy
Return
 
Launchwebsite:
RegRead,Browser,HKCR,http\shell\open\command
RegExMatch(Browser,"(?<="").+?(?="")",Browser)
Run,%Browser% http://blog.csdn.net/liuyukuan/
Return
 
SendMail:
Run mailto:1576157@qq.com
Return
 
;如果从菜单点击退出则退出。
Exit:
if WinExist(target)
{
    WinShow
    WinActivate
    MsgBox, 67,退出, 点击【是】 退出本工具同时退出%theExe%程序`n`n点击【否】仅退出本工具
    IfMsgBox Yes
    {
        WinClose
        ; 等待程序关闭
        WinWaitClose, %target%,, 5
        ExitApp
    }
    IfMsgBox Cancel
    {
        ; 用户取消，直接退出脚本
        ExitApp
    }
    IfMsgBox No
    {
        ; 用户选择不退出程序，只退出脚本
        ExitApp
    }
}
else
{
    ; 程序窗口不存在，直接退出脚本
    ExitApp
}

; 确保脚本退出
ExitApp
 
Reload:
	theWin:=getWin(target)
	WinShow %theWin%
	WinActivate %theWin%
	reload
return
 
runObsidian:
If not WinExist(target)
  run %app%
  if  WinExist(target)
  {
    WinShow
    WinActivate
  }
Return
 
ShowOrHide(WinId)
{
 
  if DllCall("IsWindowVisible", "Ptr",WinId) 
  {
    WinHide ahk_id %WinId%
  }
  else
  {
    WinShow ahk_id %WinId%
    WinActivate ahk_id %WinId%
  }
 
;	theWin:=getWin(target)
;	WinGet, OutputVar, MinMax, %theWin%
;	msgbox % OutputVar
;	if(OutputVar=-1) ;窗口处于最小化状态
;	{
;		WinHide %theWin%
;	}
;	else 
;	{
;		WinShow %theWin%	
;		winactivate %theWin%	
;	}
	return
}
MinOrRestore(target)
{
	theWin:=getWin(target)
	WinGet, OutputVar, MinMax, %theWin%
	if(OutputVar=-1) ;窗口处于最小化状态
	{
		WinRestore %theWin%
		winactivate %theWin%
		}
	else 
		WinMiniMize %theWin%	
	return
}
getWin(target)
{
	static cnt := 0	; 保持局部的计数(它的值在多次调用期间是记住的).
	static theWin
	static bk_theWin
	cnt+= 1  
	if (cnt&0x1) ;奇数判断
	{	
		WinGetTitle,theWin,%target%
		if(theWin="")
			theWin:=bk_theWin
		else
			bk_theWin:=theWin
	}
	return theWin
}