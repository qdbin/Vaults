/*
作者微信：sunwind1576157
最新版本：https://blog.csdn.net/liuyukuan/article/details/120558428
功能说明：
1、为Obsidian增加关闭到托盘功能
2、增加托盘图标，左键点击可以显示/隐藏窗口
3、可以用Win+z热键恢复显示或隐藏窗口
注意，须配置下面的app为您Obsidian本地路径
*/
SetTitleMatchMode,2
DetectHiddenWindows On
#Persistent
#SingleInstance force
CoordMode,Mouse,Screen
 
;请修改路径为您本地的obsidian程序路径
app:="C:\Users\hanbin\scoop\apps\obsidian\current\Obsidian.exe"
 
target:="Obsidian ahk_class Chrome_WidgetWin_1"
 
Menu,Tray,Icon,%app%
Menu,Tray,NoStandard
Menu,Tray,add,显示(&O),open
Menu,Tray,add,关于(&A),about
Menu,Tray,add,
 
Menu,Tray,add,退出(&E),Exit
Menu,Tray,Tip,Obsidian助手
; Menu,Tray,add,设置(&S),setting
 
Menu, Tray, Default, 显示(&O)
Menu, Tray, Click, 1
 
 
gosub open
 
 
 
 
 
SysGet,SM_CXSIZEFRAME,32
SysGet,SM_CYSIZEFRAME,33
SysGet,SM_CXSIZE,30
SysGet,SM_CYSIZE,31
Loop
{
  Sleep,100
  MouseGetPos,mx,my,win
  WinGetPos,x,y,w,h,ahk_id %win%
  l:=x+w-SM_CXSIZEFRAME-SM_CXSIZE
  t:=y ;+SM_CYSIZEFRAME
  r:=x+w-SM_CXSIZEFRAME
  b:=y+SM_CYSIZE ;+SM_CYSIZEFRAME
  If (mx<l Or mx>r Or my<t Or my>b)
  {
    Hotkey,LButton,CLICK,Off
    ToolTip
    }
  Else
  {
    WinGet,program,ProcessName,ahk_id %win%  
    if (Program<>"obsidian.exe")
            Continue
    Hotkey,LButton,CLICK,On 
    ToolTip,最小化到托盘
  }
}
Return
 
Click:
WinHide,%target%
Return
 
 
#If WinExist(target)
#o::
  ShowOrHide(target)
  Return
#if
 
 
open:
If not WinExist(target)
  run %app%
ShowOrHide(target)
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
 
 
 
Exit:
; 显示并激活Obsidian窗口
if WinExist(target)
{
    WinShow
    WinActivate
    MsgBox, 3,, 是否退出Obsidian程序?
    IfMsgBox Yes
    {
        WinClose
        ; 等待Obsidian关闭
        WinWaitClose, %target%,, 5
    }
    IfMsgBox Cancel
    {
        ; 用户取消，直接退出脚本
        ExitApp
    }
    IfMsgBox No
    {
        ; 用户选择不退出Obsidian，只退出脚本
        ExitApp
    }
}
else
{
    ; Obsidian窗口不存在，直接退出脚本
    ExitApp
}

; 确保脚本退出
ExitApp
 
ShowOrHide(target)
{
  if DllCall("IsWindowVisible", "Ptr", WinExist(target)) 
  {
    WinHide
  }
  else
  {
    WinShow
    WinActivate
  }
}