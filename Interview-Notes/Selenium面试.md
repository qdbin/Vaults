# [Selenium面试问答题](https://www.nowcoder.com/discuss/769975354977947648?sourceSSR=users)                      

###  1、什么是测试自动化或自动化测试？

​        自动化测试使用自动化工具来编写和执行测试用例，执行自动化测试套件不需要人工参与。测试人员更喜欢自动化工具来编写测试脚本和测试用例，然后组合成测试套件。自动化测试工具可以访问测试数据，控制测试的执行并将实际结果与预期结果进行比较。因此，生成被测系统的详细测试报告。

###  2、自动化测试的优势是什么？

​        自动化测试可以执行应用程序的功能和性能测试，支持重复测试用例的执行(回归测试)，并发执行，有主要测试大型测试矩阵，提高测试的准确性，节省了时间和时间。

###  3、用于功能自动化的常用自动化测试工具有哪些？

- 由 Teleric 开发的Teleric Test Studio。  
- TestingWhiz  
- HPE Unified Functional Testing (HP - UFT 以前的 QTP)  
- Tosca Testsuite  
- Watir  
- Quick Test Professional(由 HP 提供)。  
- Rational Robot(由 IBM 提供)。  
- Coded UI(由 Microsoft 提供)。  
- Selenium(开源)。
- Auto It(开源)。

###  4、用于非功能自动化的常用自动化测试工具有哪些？

- Load Runner，由 Hp 提供。
- JMeter，由 Apache 提供。
- Burp Suite，由 PortSwigger 提供。
- Acunetix，由 Acunetix 提供

###  5、什么是Selenium？

​        Selenium 是使用最广泛的开源 Web UI(用户界面)自动化测试套件之一。Jason Huggins 于 2004 年开发了 Selenium，作为 Thought Works 的内部工具。Selenium 支持跨不同浏览器、平台和编程语言的自动化。

###  6、Selenium有哪些不同的组成部分？

Selenium 不仅仅是一个工具，而是一套软件，每个软件都有不同的方法来支持自动化测试。它由四个主要组成部分组成，其中包括：

- Selenium 集成开发环境 (IDE)
- Selenium Remote Control(现已弃用)
- WebDriver
- Selenium Grid

###  7、Selenium支持哪些编程语言、浏览器和操作系统？

Selenium 支持各种操作系统、浏览器和编程语言。分别如下所示：

- 编程语言：C#、Java、Python、PHP、Ruby、Perl、JavaScript。
- 操作系统：Android、iOS、Windows、Linux、Mac、Solaris。
- 浏览器：Google Chrome、Mozilla Firefox、Internet Explorer、Edge、Opera、Safari 等。

###  8、Selenium版本有哪些重大变化/升级？

Selenium v1.0：

- 1.0 版是 Selenium 的初始版本。
- 它包括三个工具：Selenium IDE、Selenium RC 和 Selenium Grid。

Selenium v2.0：

- Selenium WebDriver 在“2.0”版本中取代了 Selenium RC。
- 随着 WebDriver 的出现，RC 被弃用并移至遗留包。

Selenium v3：

- 最新版本的 Selenium 3 具有新的附加特性和功能。
- 它包括 Selenium IDE、Selenium WebDriver 和 Selenium Grid。

###  9、Selenium支持哪些测试类型？

​        功能测试，回归测试，健全性测试，冒烟测试，响应式测试，跨浏览器测试，UI测试，集成测试

###  10、Selenium IDE是什么？

​    Selenium IDE 是作为 Firefox 扩展实现的，它在测试脚本上提供记录和回放功能。它允许测试人员以多种语言导出录制的脚本，例如 HTML、Java、Ruby、RSpec、Python、C#、JUnit 和 TestNG。Selenium IDE 的范围有限，生成的测试脚本不是很健壮且可移植。

###  11、Selenium Selenese是什么？

​        Selenium Selenese 是 Selenium 早期（1.x 时代）定义的一种基于 HTML 表格的简单脚本语言，用于编写浏览器自动化测试步骤。它主要通过 Selenium Core 和旧版 Selenium IDE 使用。虽然它易于入门，但由于缺乏编程灵活性、维护困难以及同源策略限制等缺点，它已被更强大的 Selenium WebDriver 结合主流编程语言（如 Java, Python, C#）所取代。

###  12、在Selenium中定位Web元素有哪些方法？

​        定位方法：id，class name，name，tag name，link text，partial link text，xpath，css selector

###  13、Selenium中有多少种类型的WebDriver API可用？

​        用于自动化浏览器的 WebDriver API 列表包括：AndroidDriver，ChromeDriver，EventFiringWebDriver，FirefoxDriver，HtmlUnitDriver，InternetExplorerDriver，iPhoneDriver，iPhoneSimulatorDriver，RemoteWebDriver

###  14、可以与Selenium集成以实现持续测试的自动化工具有哪些？

​    可以与Maven、Jenkins、&Docker 等自动化测试工具集成，实现持续测试；可以与 TestNG、&JUnit 等工具集成，用于管理测试用例和生成报告

###  15、Selenium中的断言是什么？

​    断言用作验证点。它**验证应用程序的状态是否符合预期**。断言的类型是：“assert”、“verify”和“waitFor”。

###  16、断言和验证命令的区别是什么？

​    **断言**：断言命令检查给定条件是真还是假。如果条件为真，程序控制将执行下一阶段的测试，如果条件为假，则停止执行，不执行任何操作。

​    **验证命令**：验证命令还检查给定条件是真还是假。它不会停止程序执行，即验证过程中的任何失败都不会停止执行，所有测试阶段都会执行。

###  17、XPath是什么？

​        XPath 是一种用于在 XML 文档中定位节点的语言。由路径表达式和一些条件组成。

###  18、XPath Absolute和XPath属性是什么？

​        XPath Absolute:

- XPath Absolute 使用户能够提及从根 HTML 标记到特定元素的完整 XPath 位置。
- 语法: //html/body/tag1[index]/tag2[index]/…/tagN[index]
- 示例: //html/body/div[2]/div/div[2]/div/div/div/fieldset/form/div[1]/input[1]

​        XPath 属性:

- 当没有适合要定位的元素的 id 或 name 属性时，始终建议使用 XPath 属性。
- 语法: //htmltag[@attribute1=’value1’ and @attribute2=’value2’]
- 示例: //input[@id=’passwd’ and @placeholder=’password’]

###  19、XPath中“/”和“//”有什么区别？

​        /：具有**绝对路径**的XPATH

​        //：具有**相对路径**的XPATH

###  20、键入键和键入命令有什么区别？

​      TypeKeys() 会触发 JavaScript 事件，而 .type() 不会

###  21、“type”和“typeAndWait”命令有什么区别？

​      “type”命令用于在软件 Web 应用程序的文本框中键入键盘键值。它也可以用于选择组合框的值，而“typeAndWait”命令在您的输入完成并且软件网页开始重新加载时使用。此命令将等待软件应用程序页面重新加载。如果输入时没有页面重新加载事件，则必须使用简单的“type”命令。

###  22、findElement()和findElements()有什么区别？

​       使用findElement选择的是符合条件的**第一个**元素， 如果没有符合条件的元素， 抛出异常

​      使用 findElements选择的是符合条件的**所有**元素， 如果没有符合条件的元素， 返回空列表

###  23、Selenium中有多少种等待类型？

​        一共有两种等待，**显性等待**和**隐性等待**

[复制代码](Selenium面试.md#)

```
import` `org.openqa.selenium.WebDriver;``import` `org.openqa.selenium.chrome.ChromeDriver;``import` `org.openqa.selenium.support.ui.*;``import` `java.time.Duration;` `WebDriver driver = ``new` `ChromeDriver();` `// 设置全局隐性等待10秒``driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(``10``));` `WebDriverWait wait = ``new` `WebDriverWait(driver, Duration.ofSeconds(``10``));``// 显性等待元素可见``WebElement element = wait.until(ExpectedConditions.visibilityOfElementLocated(By.id(``"elementId"``)));``// 等待元素可点击``wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector(``"button"``))).click();``// 等待文本出现``wait.until(ExpectedConditions.textToBePresentInElementLocated(By.id(``"msg"``), ``"成功"``));
```

[复制代码](Selenium面试.md#)

```
from selenium.webdriver.support.ui ``import` `WebDriverWait``from selenium.webdriver.support ``import` `expected_conditions as EC``from selenium.webdriver.common.by ``import` `By``from selenium ``import` `webdriver` `driver = webdriver.Chrome()``# 设置全局隐性等待``10``秒``driver.implicitly_wait(``10``)` `wait = WebDriverWait(driver, ``10``)``# 等待元素可见``element = wait.until(EC.visibility_of_element_located((By.ID, ``"elementId"``)))``# 等待元素可点击``wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ``"button"``))).click()``# 等待文本出现``wait.until(EC.text_to_be_present_in_element((By.ID, ``"msg"``), ``"成功"``))
```

### 24、Selenium中隐式等待的主要缺点是什么？

主要测试缺点：降低测试的性能；同时，如果在设定的等待时间内没有等到元素的出现，测试用例失败

###  25、Selenium Grid/网格是什么？

​        Selenium Grid 是 Selenium 套件中的一个关键组件，专门设计用于在多个机器（物理机或虚拟机）和不同的操作系统、浏览器上并行运行 WebDriver 测试脚本。 它的核心目的是实现分布式测试执行和跨环境测试。Selenium Grid方便将测试分布在多台机器上，并且同时分布在所有机器上。因此，可以使用相同的文本脚本在 Windows 上的 Internet Explorer 和 Mac 机器上的 Safari 上执行测试。它减少了测试执行的时间并提供了快速的反馈。

###  26、如何在Selenium WebDriver中启动不同的浏览器？

[复制代码](Selenium面试.md#)

```
import` `org.openqa.selenium.WebDriver;``import` `org.openqa.selenium.safari.SafariDriver;` `public` `class` `LaunchSafari {``  ``public` `static` `void` `main(String[] args) {``    ``// 无需指定驱动路径（系统内置）``    ``// 需在 Safari 设置中启用开发者模式：``    ``// Safari > 偏好设置 > 高级 > 勾选"在菜单栏中显示开发菜单"``    ``WebDriver driver = ``new` `SafariDriver();``    ``driver.get(``"https://www.google.com"``);``    ``System.out.println(``"Safari 标题: "` `+ driver.getTitle());``    ``driver.quit();``  ``}``}
```

[复制代码](Selenium面试.md#)

```
import` `org.openqa.selenium.WebDriver;``import` `org.openqa.selenium.edge.EdgeDriver;``import` `org.openqa.selenium.edge.EdgeOptions;` `public` `class` `LaunchEdge {``  ``public` `static` `void` `main(String[] args) {``    ``System.setProperty(``"webdriver.edge.driver"``, ``"path/to/msedgedriver"``);   ``    ``EdgeOptions options = ``new` `EdgeOptions();``    ``options.addArguments(``"--inprivate"``); ``// 无痕模式    ``    ``WebDriver driver = ``new` `EdgeDriver(options);``    ``driver.get(``"https://www.google.com"``);``    ``System.out.println(``"Edge 标题: "` `+ driver.getTitle());``    ``driver.quit();``  ``}``}
```

[复制代码](Selenium面试.md#)

```
import` `org.openqa.selenium.WebDriver;``import` `org.openqa.selenium.firefox.FirefoxDriver;``import` `org.openqa.selenium.firefox.FirefoxOptions;` `public` `class` `LaunchFirefox {``  ``public` `static` `void` `main(String[] args) {``    ``System.setProperty(``"webdriver.gecko.driver"``, ``"path/to/geckodriver"``);``    ``FirefoxOptions options = ``new` `FirefoxOptions();``    ``options.addArguments(``"--headless"``);    ``    ``WebDriver driver = ``new` `FirefoxDriver(options);``    ``driver.get(``"https://www.google.com"``);``    ``System.out.println(``"Firefox 标题: "` `+ driver.getTitle());``    ``driver.quit();``  ``}``}
```

[复制代码](Selenium面试.md#)

```
import` `org.openqa.selenium.WebDriver;``import` `org.openqa.selenium.chrome.ChromeDriver;``import` `org.openqa.selenium.chrome.ChromeOptions;` `public` `class` `LaunchChrome {``  ``public` `static` `void` `main(String[] args) {``    ``// 指定驱动路径（如果未添加到PATH）``    ``System.setProperty(``"webdriver.chrome.driver"``, ``"path/to/chromedriver"``);   ``    ``// 可选：添加配置选项（如无头模式）``    ``ChromeOptions options = ``new` `ChromeOptions();``    ``options.addArguments(``"--headless"``); ``// 无头模式``    ``options.addArguments(``"--start-maximized"``); ``// 最大化窗口    ``    ``// 初始化 Chrome 驱动``    ``WebDriver driver = ``new` `ChromeDriver(options);    ``    ``// 打开网页``    ``driver.get(``"https://www.google.com"``);``    ``System.out.println(``"Chrome 标题: "` `+ driver.getTitle());    ``    ``// 关闭浏览器``    ``driver.quit();``  ``}``}
```

###  27、请编写代码片段以在WebDriver中启动Chrome浏览器？

[复制代码](Selenium面试.md#)

```
public` `class` `ChromeBrowserLaunchDemo { ` `  ``public` `static` `void` `main(String[] args) {  ``    ``WebDriver driver; ``    ``System.setProperty(``"webdriver.chrome.driver"``, ``"/lib/chromeDriver/chromedriver.exe"``); ``    ``driver = newChromeDriver();  ``    ``driver.get(``"https://www.baidu.com"``); ``    ``driver.quit(); ``  ``}``}
```

###  28、编写代码片段以在WebDriver中执行右键单击元素？

[复制代码](Selenium面试.md#)

```
Actions action = newActions(driver); ``WebElement element = driver.findElement(By.id(``"elementId"``)); ``action.contextClick(element).perform();
```

###  29、编写代码片段以在WebDriver中执行鼠标悬停？

[复制代码](Selenium面试.md#)

```
Actions action = newActions(driver); ``WebElement element = driver.findElement(By.id(``"elementId"``)); ``action.moveToElement(element).perform();
```

###  30、在WebDriver中如何进行拖放操作？

[复制代码](Selenium面试.md#)

```
fromWebElement = driver.findElement(By Locator of fromWebElement);  ``toWebElement = driver.findElement(By Locator of toWebElement);  ``Actions builder = newActions(driver); ``Action dragAndDrop = builder.clickAndHold(fromWebElement) ``       ``.moveToElement(toWebElement) ``       ``.release(toWebElement) ``       ``.build(); ``dragAndDrop.perform();
```

###  31、在WebDriver中刷新网页有哪些方法？

[复制代码](Selenium面试.md#)

```
import` `org.openqa.selenium.By;``import` `org.openqa.selenium.Keys;``import` `org.openqa.selenium.WebDriver;``import` `org.openqa.selenium.WebElement;``import` `org.openqa.selenium.chrome.ChromeDriver;` `public` `class` `PageRefreshMethods {``  ``public` `static` `void` `main(String[] args) {``    ``// 设置驱动路径``    ``System.setProperty(``"webdriver.chrome.driver"``, ``"path/to/chromedriver"``);``    ``WebDriver driver = ``new` `ChromeDriver();` `    ``try` `{``      ``// 打开测试页面``      ``driver.get(``"https://example.com"``);` `      ``// 方法1: navigate().refresh()``      ``driver.navigate().refresh();``      ``System.out.println(``"Refreshed via Method 1"``);` `      ``// 方法2: getCurrentUrl() + get()``      ``String url = driver.getCurrentUrl();``      ``driver.get(url);``      ``System.out.println(``"Refreshed via Method 2"``);` `      ``// 方法3: getCurrentUrl() + navigate().to()``      ``url = driver.getCurrentUrl();``      ``driver.navigate().to(url);``      ``System.out.println(``"Refreshed via Method 3"``);` `      ``// 方法4: sendKeys(Keys.F5)``      ``WebElement body = driver.findElement(By.tagName(``"body"``));``      ``body.sendKeys(Keys.F5);``      ``System.out.println(``"Refreshed via Method 4"``);` `      ``// 方法5: sendKeys("\uE035")``      ``body = driver.findElement(By.tagName(``"body"``));``      ``body.sendKeys(``"\uE035"``); ``// Unicode for F5``      ``System.out.println(``"Refreshed via Method 5"``);` `    ``} ``finally` `{``      ``driver.quit();``    ``}``  ``}``}
```

###  32、编写代码在浏览器历史记录中前后导航？

[复制代码](Selenium面试.md#)

```
import` `org.openqa.selenium.WebDriver;``import` `org.openqa.selenium.chrome.ChromeDriver;` `public` `class` `BrowserHistoryNavigation {``  ``public` `static` `void` `main(String[] args) {``    ``// 设置 ChromeDriver 路径（根据你的实际路径修改）``    ``System.setProperty(``"webdriver.chrome.driver"``, ``"path/to/chromedriver"``);``    ` `    ``WebDriver driver = ``new` `ChromeDriver();` `    ``try` `{``      ``// 访问第一个页面``      ``driver.get(``"https://www.example.com/page1"``);``      ``Thread.sleep(``2000``); ``// 等待2秒（仅为演示效果）` `      ``// 访问第二个页面``      ``driver.get(``"https://www.example.com/page2"``);``      ``Thread.sleep(``2000``);` `      ``// 后退到历史记录中的上一个页面（返回 page1）``      ``driver.navigate().back();``      ``System.out.println(``"后退到: "` `+ driver.getCurrentUrl());``      ``Thread.sleep(``2000``);` `      ``// 前进到历史记录中的下一个页面（返回 page2）``      ``driver.navigate().forward();``      ``System.out.println(``"前进到: "` `+ driver.getCurrentUrl());``      ``Thread.sleep(``2000``);` `      ``// 刷新当前页面``      ``driver.navigate().refresh();``      ``System.out.println(``"已刷新页面"``);` `    ``} ``catch` `(InterruptedException e) {``      ``e.printStackTrace();``    ``} ``finally` `{``      ``driver.quit(); ``// 关闭浏览器``    ``}``  ``}``}
```

###  33、怎样才能得到一个网页元素的文本？

[复制代码](Selenium面试.md#)

```
WebElement ele =driver.findElement(By.id(``"elementId"``)); `` ``if` `(ele != ``null``){`` ``System.out.println(``"Text():"``+ ele.getText());`` ``System.out.println(``"Value:"``+ ele.getAttribute(``"Value"``));`` ``System.out.println(``"innerText():"``+ ele.getAttribute(``"innerText"``)); `` ``System.out.printIn(``"textContent():"``+ ele.getAttribute(``"textContent"``));`` ``System.out.println(``"isEnable():"``+ ele.isEnabled());`` ``System.out.println(``"aria-disabled():"``+ ele.getAttribute(``"aria-disabled"``));`` ``System.out.printIn(``"innerHTML():"``+ ele.getAttribute(``"innerHTML"``));`` ``System.out.printIn(``"outerHTML():"``+ ele.getAttribute(``"outerHTML"``));`` ``}
```

###  34、如何在下拉列表中选择值？

[复制代码](Selenium面试.md#)

```
import` `org.openqa.selenium.WebDriver;``import` `org.openqa.selenium.chrome.ChromeDriver;``import` `org.openqa.selenium.By;``import` `org.openqa.selenium.WebElement;``import` `org.openqa.selenium.support.ui.Select;` `public` `class` `DropdownExample {``  ``public` `static` `void` `main(String[] args) {``    ``// 设置 WebDriver``    ``System.setProperty(``"webdriver.chrome.driver"``, ``"path/to/chromedriver"``);``    ``WebDriver driver = ``new` `ChromeDriver();``    ` `    ``// 打开页面``    ``driver.get(``"https://example.com/page-with-dropdown"``);``    ` `    ``// 定位下拉列表``    ``WebElement dropdown = driver.findElement(By.id(``"country"``));``    ``Select select = ``new` `Select(dropdown);``    ` `    ``// 选择值（三种方式任选其一）``    ``select.selectByVisibleText(``"Brazil"``);   ``// 按文本选择``    ``select.selectByValue(``"br"``);      ``// 按 value 属性选择``    ``select.selectByIndex(``5``);        ``// 按索引选择（索引从0开始）``    ` `    ``// 关闭浏览器``    ``driver.quit();``  ``}``}
```

###  35、有哪些不同类型的导航命令？

- navigate().to()，等效操作：driver.get()	
- navigate().back()，表示浏览器后退按钮，依赖浏览历史记录
- navigate().forward()，表示浏览器前进按钮，需先执行过 back()
- navigate().refresh()，表示F5 / 刷新按钮，重新加载当前页

###  36、如何处理WebDriver中的框架？

[复制代码](Selenium面试.md#)

```
//1.通过索引切换 (Index)``// 切换到第一个 iframe``driver.switchTo().frame(``0``);``// 操作 iframe 内的元素``driver.findElement(By.id(``"innerElement"``)).click();``// 切回主页面``driver.switchTo().defaultContent();` `// 2.通过 ID 切换``driver.switchTo().frame(``"iframe-id"``);` `// 3.通过 Name 切换``driver.switchTo().frame(``"iframe-name"``);` `// 4.通过 CSS 选择器定位``WebElement iframeElement = driver.findElement(By.cssSelector(``"iframe.selector-class"``));``driver.switchTo().frame(iframeElement);` `// 5.通过 XPath 定位``WebElement iframeElement = driver.findElement(By.xpath(``"//iframe[@attribute='value']"``));``driver.switchTo().frame(iframeElement);` `// 6.操作 iframe 内部元素``driver.switchTo().frame(``"inner-frame"``);``driver.findElement(By.id(``"button"``)).click();``// 返回父级 iframe（嵌套 iframe 时使用）``driver.switchTo().parentFrame();``// 直接返回主页面``driver.switchTo().defaultContent();` `//7.多层 iframe 嵌套操作``// 切换到父级 iframe``driver.switchTo().frame(``"parent"``);``// 切换到子级 iframe``WebElement childFrame = driver.findElement(By.id(``"child"``));``driver.switchTo().frame(childFrame);``// 在子级 iframe 中操作``driver.findElement(By.id(``"child-element"``)).click();``// 返回父级 iframe``driver.switchTo().parentFrame();``// 在父级 iframe 中操作``driver.findElement(By.id(``"parent-element"``)).click();``// 返回主页面``driver.switchTo().defaultContent();
```

###  37、.NET是否有HtmlUnitDriver？

- HtmlUnitDriver 是一个基于 Java 的无界面浏览器（Headless Browser），使用纯 Java 实现（无需真实浏览器）。
- 它属于 Selenium Java 客户端库的一部分，但 .NET 版的 Selenium 官方包（Selenium.WebDriver）不包含类似组件。



​          .NET 中的替代方案

[复制代码](Selenium面试.md#)

```
//方案 1：使用无头模式（Headless Mode）的真实浏览器``// Chrome 无头模式示例``var options = ``new` `ChromeOptions();``options.AddArgument(``"--headless"``); ``// 启用无头模式``IWebDriver driver = ``new` `ChromeDriver(options);` `//方案 2：使用 .NET 的轻量级 HTML 解析库``//HtmlAgilityPack：流行的 HTML 解析库：``var web = ``new` `HtmlWeb();``var doc = web.Load(``"https://example.com"``);``var node = doc.DocumentNode.SelectSingleNode(``"//h1"``);` `//AngleSharp：支持更复杂的 DOM 操作和基础 JavaScript：``var context = BrowsingContext.New(Configuration.Default);``var document = await context.OpenAsync(``"https://example.com"``);``var element = document.QuerySelector(``"h1"``);
```

###  38、如何通过某些代理从浏览器重定向浏览？

[复制代码](Selenium面试.md#)

```
import` `org.openqa.selenium.Proxy;``import` `org.openqa.selenium.WebDriver;``import` `org.openqa.selenium.chrome.ChromeDriver;``import` `org.openqa.selenium.chrome.ChromeOptions;` `public` `class` `ProxyExample {``  ``public` `static` `void` `main(String[] args) {``    ``// 1. 设置代理参数``    ``String proxyAddress = ``"127.0.0.1:8080"``; ``// 替换为你的代理IP和端口``    ``String proxyType = ``"http"``;        ``// 代理类型：http/socks``    ` `    ``// 2. 创建代理配置``    ``Proxy proxy = ``new` `Proxy();``    ``proxy.setProxyType(Proxy.ProxyType.MANUAL);``    ``proxy.setHttpProxy(proxyAddress);``    ``proxy.setSslProxy(proxyAddress);``    ``proxy.setSocksProxy(proxyAddress);    ``// 如果是SOCKS代理``    ``proxy.setSocksUsername(``"user"``);     ``// SOCKS用户名（可选）``    ``proxy.setSocksPassword(``"pass"``);     ``// SOCKS密码（可选）` `    ``// 3. 配置浏览器选项``    ``ChromeOptions options = ``new` `ChromeOptions();``    ``options.setProxy(proxy);``    ``options.addArguments(``"--ignore-certificate-errors"``); ``// 忽略证书错误` `    ``// 4. 启动带代理的浏览器``    ``System.setProperty(``"webdriver.chrome.driver"``, ``"chromedriver.exe"``);``    ``WebDriver driver = ``new` `ChromeDriver(options);``    ` `    ``// 5. 验证代理``    ``driver.get(``"https://whatismyip.com"``); // 访问IP检测网站``  ``}``}
```

###  39、什么是POM(页面对象模型)？它的优点是什么？

​        POM 是自动化测试领域的一个基石性设计模式。它通过将页面抽象为对象并封装其细节，完美地实现了关注点分离：测试脚本负责定义测试流程和验证点（What to do），Page Object 负责具体如何与页面交互（How to do it）。这种分离带来了更高的可维护性、可读性、重用性和健壮性，使得自动化测试项目在面对频繁的 UI 变更和规模增长时，能够保持高效和稳定。对于任何稍具规模或需要长期维护的 Web 自动化测试项目，采用 POM 几乎是必须的选择。

###  40、如何在WebDriver中截取屏幕截图？

[复制代码](Selenium面试.md#)

```
//方法 1：捕获整个浏览器窗口（最常用）``import` `org.apache.commons.io.FileUtils;``import` `org.openqa.selenium.OutputType;``import` `org.openqa.selenium.TakesScreenshot;``import` `org.openqa.selenium.WebDriver;``import` `java.io.File;``import` `java.io.IOException;` `public` `class` `ScreenshotUtil {``  ` `  ``public` `static` `void` `captureScreenshot(WebDriver driver, String screenshotName) {``    ``try` `{``      ``// 1. 将 WebDriver 强制转换为 TakesScreenshot 类型``      ``TakesScreenshot ts = (TakesScreenshot) driver;``      ` `      ``// 2. 调用 getScreenshotAs 方法创建图像文件``      ``File source = ts.getScreenshotAs(OutputType.FILE);``      ` `      ``// 3. 指定保存路径``      ``String destPath = ``"./screenshots/"` `+ screenshotName + ``".png"``;``      ``File destination = ``new` `File(destPath);``      ` `      ``// 4. 复制文件到目标位置``      ``FileUtils.copyFile(source, destination);``      ` `      ``System.out.println(``"截图保存至: "` `+ destPath);``    ``} ``catch` `(IOException e) {``      ``System.out.println(``"截图失败: "` `+ e.getMessage());``    ``}``  ``}``}` `//方法 2：捕获特定 WebElement``import` `org.openqa.selenium.WebElement;``import` `org.apache.commons.io.FileUtils;` `public` `static` `void` `captureElementScreenshot(WebElement element, String fileName) {``  ``try` `{``    ``// 直接通过元素获取截图``    ``File srcFile = element.getScreenshotAs(OutputType.FILE);``    ``FileUtils.copyFile(srcFile, ``new` `File(``"./element_screenshots/"` `+ fileName + ``".png"``));``  ``} ``catch` `(IOException e) {``    ``e.printStackTrace();``  ``}``}` `//方法 3：使用 Base64 编码（适用于报告嵌入）``public` `static` `String getBase64Screenshot(WebDriver driver) {``  ``return` `((TakesScreenshot) driver).getScreenshotAs(OutputType.BASE64);``}` `// 用法示例：``String base64Img = getBase64Screenshot(driver);``System.out.println(``"data:image/png;base64,"` `+ base64Img); ``// 可直接嵌入HTML报告
```

###  41、如何使用Selenium在文本框中输入文本？

[复制代码](Selenium面试.md#)

```
//1.输入``driver.findElement(By.xpath(``"//xpathExpression"``)).sendKeys(``"test"``);` `//2.输入``Actions actions = ``new` `Actions(driver);``actions.sendKeys(``"test"``).perform();` `//3.输入``// 创建一个StringSelection对象，用于存储要复制的文本内容``StringSelection selection = ``new` `StringSelection(``"文本"``);``// 获取系统剪贴板``Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();   ``// 将StringSelection对象放入剪贴板``clipboard.setContents(selection, ``null``);``// 模拟Ctrl+V操作``Robot robot = ``new` `Robot();``robot.keyPress(KeyEvent.VK_CONTROL);``robot.keyPress(KeyEvent.VK_V);``robot.keyRelease(KeyEvent.VK_V);``robot.keyRelease(KeyEvent.VK_CONTROL);
```

###  42、怎么知道一个元素是否显示在屏幕上？

​        WebDriver 允许用户检查 Web 元素的可见性。这些网络元素可以是按钮、单选按钮、下拉菜单、复选框、框、标签等，它们与以下方法一起使用。函数：isDisplayed()，isSelected()，isEnabled()

[复制代码](Selenium面试.md#)

```
public` `boolean` `isElementVisibleOnScreen(WebDriver driver, By locator) {``  ``try` `{``    ``WebElement element = driver.findElement(locator);``    ``// 检查基础可见性 && 在视口内``    ``return` `element.isDisplayed() && isElementInViewport(driver, element);``  ``} ``catch` `(NoSuchElementException e) {``    ``return` `false``; ``// 元素不存在``  ``}``}` `// 使用示例``boolean` `isVisible = isElementVisibleOnScreen(driver, By.id(``"submitBtn"``));
```

###  43、如何使用linkText点击超链接？

[复制代码](Selenium面试.md#)

```
import` `org.openqa.selenium.By;``import` `org.openqa.selenium.WebDriver;``import` `org.openqa.selenium.chrome.ChromeDriver;` `public` `class` `LinkTextExample {``  ``public` `static` `void` `main(String[] args) {``    ``// 1. 设置WebDriver路径（如果已配置环境变量可跳过）``    ``System.setProperty(``"webdriver.chrome.driver"``, ``"C:/path/to/chromedriver.exe"``);``    ` `    ``// 2. 初始化WebDriver``    ``WebDriver driver = ``new` `ChromeDriver();``    ` `    ``try` `{``      ``// 3. 打开目标网页``      ``driver.get(``"https://www.example.com"``);``      ` `      ``// 4. 使用linkText定位并点击超链接``      ``driver.findElement(By.linkText(``"Contact Us"``)).click(); ``// 精确匹配文本``      ` `      ``// 可选：添加等待确保页面加载（实际使用中建议用显式等待）``      ``Thread.sleep(``2000``);``      ` `    ``} ``catch` `(Exception e) {``      ``e.printStackTrace();``    ``} ``finally` `{``      ``// 5. 关闭浏览器``      ``driver.quit();``    ``}``  ``}``}
```

参考链接：[https://cloud.tencent.com/developer/article/2240924](https://gw-c.nowcoder.com/api/sparta/jump/link?link=https%3A%2F%2Fcloud.tencent.com%2Fdeveloper%2Farticle%2F2240924)

作者：玖拾肆
链接：https://www.nowcoder.com/discuss/769975354977947648?sourceSSR=users
来源：牛客网