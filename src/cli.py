"""
CLI主程序
交互式命令行界面 + 一键运行模式
"""

import sys
import os
from pathlib import Path
from typing import Optional, List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_toolkit import PromptSession, HTML
from prompt_toolkit.shortcuts import confirm
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import print as rprint

from src.utils.config import get_config, Config
from src.apis.llm_client import LLMClient
from src.apis.arxiv_api import ArxivAPI, ArxivPaper
from src.analyzers.paper_analyzer import PaperAnalyzer, PaperAnalysis
from src.renderers.markdown_gen import MarkdownGenerator
from src.utils.io import ensure_dir, safe_filename_from_title
from src.agents.auto_agent import AutoAgent, AgentConfig


class PaperReaderCLI:
    """PaperReader CLI应用"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化CLI

        Args:
            config_path: 配置文件路径
        """
        self.console = Console()
        self.config = get_config(config_path)
        self.session = PromptSession()

        # 初始化组件
        self.llm_client: Optional[LLMClient] = None
        self.arxiv_api: Optional[ArxivAPI] = None
        self.analyzer: Optional[PaperAnalyzer] = None
        self.generator: Optional[MarkdownGenerator] = None

        # 状态
        self.papers: List[ArxivPaper] = []
        self.analyses: List[PaperAnalysis] = []

        # 持久化参数（记忆用户上一次的选择）
        self.saved_params = {
            "max_results": self.config.search.get("max_results", 10),
            "download_pdf": True,
            "use_obsidian": True,
            "ref_depth": 0,
            "search_field": "all"  # 搜索字段: all, ti, abs
        }

    def show_welcome(self) -> None:
        """显示欢迎信息"""
        welcome_text = """
[bold cyan]╔═══════════��════════════════════════════╗
║     🤖 AI Paper Reader 📚              ║
║     学术论文智能阅读助手                     ║
╚════════════════════════════════════════╝[/bold cyan]

欢迎使用AI Paper Reader！
这个工具可以帮助您搜索、分析和总结学术论文。

输入 [bold yellow]help[/bold yellow] 查看帮助，[bold yellow]quit[/bold yellow] 退出程序。
"""

        self.console.print(Panel(welcome_text, border_style="cyan"))

    def show_help(self) -> None:
        """显示帮助信息"""
        help_table = Table(title="命令列表", show_header=True, header_style="bold magenta")
        help_table.add_column("命令", style="cyan", width=20)
        help_table.add_column("说明", style="white")

        commands = [
            ("auto [主题]", "🚀 一键自动运行（推荐）"),
            ("search", "搜索论文"),
            ("list", "列出已搜索的论文"),
            ("analyze", "分析论文"),
            ("generate", "生成索引文件"),
            ("config", "查看/修改配置"),
            ("help", "显示帮助"),
            ("quit", "退出程序")
        ]

        for cmd, desc in commands:
            help_table.add_row(cmd, desc)

        self.console.print(help_table)
        self.console.print("\n[bold yellow]推荐使用 'auto' 命令一键完成：[/bold yellow]")
        self.console.print("  AI自动拓展搜索 → 下载PDF → 分析 → 生成索引")

    def init_llm_client(self) -> bool:
        """初始化LLM客户端"""
        llm_config = self.config.llm
        api_key = llm_config.get("api_key", "")

        if not api_key:
            self.console.print("[red]错误: 未配置LLM API密钥！[/red]")
            self.console.print("请在config/config.yaml中配置api_key，或设置环境变量LLM_API_KEY")
            return False

        try:
            self.llm_client = LLMClient(
                api_key=api_key,
                base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
                model=llm_config.get("model", "gpt-3.5-turbo"),
                temperature=llm_config.get("temperature", 0.3),
                max_tokens=llm_config.get("max_tokens", 4000),
                timeout=llm_config.get("timeout", 60)
            )

            # 初始化分析器
            self.analyzer = PaperAnalyzer(
                llm_client=self.llm_client,
                analyze_full_text=self.config.analysis.get("analyze_full_text", True),
                max_pages=self.config.analysis.get("max_pages", 0)
            )

            self.console.print("[green]✓ LLM客户端初始化成功[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]✗ LLM客户端初始化失败: {e}[/red]")
            return False

    def init_arxiv_api(self, search_field: str = "all") -> None:
        """初始化arXiv API"""
        search_config = self.config.search
        self.arxiv_api = ArxivAPI(
            max_results=search_config.get("max_results", 10),
            sort_by=search_config.get("sort_by", "relevance"),
            categories=search_config.get("categories", []),
            search_field=search_field
        )

        self.console.print("[green]✓ arXiv API初始化成功[/green]")

    def cmd_search(self, query: Optional[str] = None) -> None:
        """搜索论文"""
        if query is None:
            query = self.session.prompt(HTML("<ansicyan>请输入搜索关键词:</ansicyan> "))

        if not query:
            self.console.print("[yellow]已取消搜索[/yellow]")
            return

        # 初始化arXiv API
        if self.arxiv_api is None:
            self.init_arxiv_api()

        # 询问数量
        max_results = self.config.search.get("max_results", 10)
        num_input = self.session.prompt(
            HTML(f"<ansicyan>搜索数量 (默认{max_results}):</ansicyan> "),
            default=str(max_results)
        )

        try:
            max_results = int(num_input) if num_input else max_results
        except ValueError:
            max_results = self.config.search.get("max_results", 10)

        # 搜索
        with self.console.status(f"[bold cyan]正在搜索: {query}..."):
            try:
                self.papers = self.arxiv_api.search(
                    query=query,
                    max_results=max_results
                )
            except Exception as e:
                self.console.print(f"[red]搜索失败: {e}[/red]")
                return

        # 显示结果
        if not self.papers:
            self.console.print("[yellow]未找到相关论文[/yellow]")
            return

        self.console.print(f"[green]找到 {len(self.papers)} 篇论文[/green]\n")
        self._show_papers_list(self.papers)

    def _show_papers_list(self, papers: List[ArxivPaper], limit: int = 5) -> None:
        """显示论文列表"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="cyan", width=4)
        table.add_column("标题", style="white", width=50)
        table.add_column("作者", style="yellow", width=20)
        table.add_column("发布日期", style="green", width=12)

        for i, paper in enumerate(papers[:limit], 1):
            title = paper.title[:47] + "..." if len(paper.title) > 50 else paper.title
            authors = paper.authors_str[:17] + "..." if len(paper.authors_str) > 20 else paper.authors_str

            table.add_row(
                str(i),
                title,
                authors,
                paper.published.strftime("%Y-%m-%d")
            )

        self.console.print(table)

        if len(papers) > limit:
            self.console.print(f"[dim]... 还有 {len(papers) - limit} 篇论文[/dim]")

    def cmd_list(self) -> None:
        """列出已搜索的论文"""
        if not self.papers:
            self.console.print("[yellow]暂无论文记录，请先使用search命令搜索[/yellow]")
            return

        self.console.print(f"\n[bold]共 {len(self.papers)} 篇论文[/bold]\n")
        self._show_papers_list(self.papers, limit=10)

    def cmd_analyze(self) -> None:
        """分析论文"""
        if not self.papers:
            self.console.print("[yellow]请先使用search命令搜索论文[/yellow]")
            return

        # 初始化LLM客户端
        if self.analyzer is None:
            if not self.init_llm_client():
                return

        # 询问是否下载PDF
        download_pdf = confirm("是否下载PDF进行完整分析？")
        papers_dir = ensure_dir(self.config.output.get("papers_dir", "./data/papers"))

        # 分析进度
        self.analyses = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:

            task = progress.add_task("[cyan]分析论文中...", total=len(self.papers))

            for i, paper in enumerate(self.papers, 1):
                progress.update(task, description=f"[cyan]分析第 {i}/{len(self.papers)} 篇: {paper.title[:30]}...")

                pdf_path = None
                if download_pdf:
                    # 下载PDF
                    filename = safe_filename_from_title(f"{paper.arxiv_id}_{paper.title}")
                    pdf_path = papers_dir / f"{filename}.pdf"

                    if not pdf_path.exists():
                        try:
                            self.arxiv_api.download_pdf(paper, str(pdf_path))
                        except Exception as e:
                            self.console.print(f"[yellow]下载PDF失败: {e}，仅分析摘要[/yellow]")
                            pdf_path = None

                # 分析
                try:
                    analysis = self.analyzer.analyze_from_arxiv(paper, str(pdf_path) if pdf_path else None)
                    self.analyses.append(analysis)
                except Exception as e:
                    self.console.print(f"[red]分析失败: {e}[/red]")

                progress.advance(task)

        self.console.print(f"\n[green]✓ 分析完成！共分析 {len(self.analyses)} 篇论文[/green]")

    def cmd_generate(self) -> None:
        """生成索引文件"""
        if not self.analyses:
            self.console.print("[yellow]请先使用analyze命令分析论文[/yellow]")
            return

        # 初始化生成器
        if self.generator is None:
            self.generator = MarkdownGenerator(
                include_full_summary=self.config.output.get("include_full_summary", True)
            )

        # 生成索引
        index_file = self.config.output.get("index_file", "./output/papers_index.md")

        with self.console.status("[bold cyan]生成索引文件..."):
            content = self.generator.generate_index(
                self.analyses,
                title="论文索引",
                metadata={"搜索关键词": "..."}
            )

            # 保存
            self.generator.save_to_file(content, index_file)

        self.console.print(f"[green]✓ 索引文件已生成: {index_file}[/green]")

    def cmd_config(self) -> None:
        """查看配置"""
        table = Table(title="当前配置", show_header=True)
        table.add_column("配置项", style="cyan", width=25)
        table.add_column("值", style="yellow")

        # LLM配置
        llm = self.config.llm
        table.add_row("LLM Model", llm.get("model", "N/A"))
        table.add_row("LLM Base URL", llm.get("base_url", "N/A"))
        table.add_row("LLM API Key", "***" if llm.get("api_key") else "未配置")

        # 搜索配置
        search = self.config.search
        table.add_row("最大结果数", str(search.get("max_results", 10)))
        table.add_row("排序方式", search.get("sort_by", "relevance"))
        table.add_row("分类", ", ".join(search.get("categories", [])))

        # 输出配置
        output = self.config.output
        table.add_row("PDF目录", output.get("papers_dir", "./data/papers"))
        table.add_row("索引文件", output.get("index_file", "./output/papers_index.md"))

        self.console.print(table)

    def cmd_auto(self, topic: Optional[str] = None) -> None:
        """
        一键式自动运行
        AI自动搜索→下载→分析→生成索引
        """
        if topic is None:
            topic = self.session.prompt(HTML("<ansicyan>请输入研究主题:</ansicyan> "))

        if not topic:
            self.console.print("[yellow]已取消[/yellow]")
            return

        # 询问论文数量（使用上次的值作为默认）
        max_results_input = self.session.prompt(
            HTML(f"<ansicyan>论文数量 (默认{self.saved_params['max_results']}):</ansicyan> "),
            default=str(self.saved_params["max_results"])
        )

        try:
            max_results = int(max_results_input) if max_results_input else self.saved_params["max_results"]
            self.saved_params["max_results"] = max_results
        except ValueError:
            max_results = self.saved_params["max_results"]

        # 询问是否下载PDF进行完整分析（使用上次的值作为默认）
        default_pdf = "Y" if self.saved_params["download_pdf"] else "n"
        download_pdf_input = self.session.prompt(
            HTML(f"<ansicyan>是否下载PDF进行完整分析? [{'Y/n' if self.saved_params['download_pdf'] else 'y/N'}]:</ansicyan> "),
            default=default_pdf
        )
        download_pdf = download_pdf_input.lower() not in ['n', 'no']
        self.saved_params["download_pdf"] = download_pdf

        # 询问是否生成Obsidian知识库（使用上次的值作为默认）
        default_obsidian = "Y" if self.saved_params["use_obsidian"] else "n"
        use_obsidian_input = self.session.prompt(
            HTML(f"<ansicyan>是否生成Obsidian知识库? [{'Y/n' if self.saved_params['use_obsidian'] else 'y/N'}]:</ansicyan> "),
            default=default_obsidian
        )
        use_obsidian = use_obsidian_input.lower() not in ['n', 'no']
        self.saved_params["use_obsidian"] = use_obsidian

        # 询问参考文献挖掘层级（使用上次的值作为默认）
        ref_depth_input = self.session.prompt(
            HTML(f"<ansicyan>参考文献挖掘层级 (0=不挖掘, 1-3=递归层级, 默认{self.saved_params['ref_depth']}):</ansicyan> "),
            default=str(self.saved_params["ref_depth"])
        )

        try:
            ref_depth = int(ref_depth_input) if ref_depth_input else self.saved_params["ref_depth"]
            ref_depth = max(0, min(3, ref_depth))  # 限制在0-3之间
            self.saved_params["ref_depth"] = ref_depth
        except ValueError:
            ref_depth = self.saved_params["ref_depth"]

        if ref_depth > 0:
            self.console.print(f"[cyan]将进行 {ref_depth} 层参考文献挖掘[/cyan]")

        # 询问搜索字段（使用上次的值作为默认）
        field_names = {
            "all": "所有字段 (推荐)",
            "ti": "标题",
            "abs": "摘要"
        }
        field_input = self.session.prompt(
            HTML(f"<ansicyan>搜索字段 (all=所有字段, ti=标题, abs=摘要, 默认{self.saved_params['search_field']}):</ansicyan> "),
            default=self.saved_params["search_field"]
        )

        if field_input in ["all", "ti", "abs"]:
            search_field = field_input
            self.saved_params["search_field"] = search_field
            self.console.print(f"[cyan]搜索字段: {field_names.get(search_field, search_field)}[/cyan]")
        else:
            search_field = self.saved_params["search_field"]

        # 初始化LLM客户端
        if self.llm_client is None:
            if not self.init_llm_client():
                return

        # 初始化arXiv API
        if self.arxiv_api is None:
            self.init_arxiv_api(search_field)
        else:
            # 更新现有API的搜索字段
            self.arxiv_api.search_field = search_field

        # 使用AutoAgent
        from src.agents.auto_agent import AutoAgent, AgentConfig

        agent_config = AgentConfig(
            max_papers=max_results,
            download_pdfs=download_pdf,
            analyze_full_text=download_pdf,
            expand_references=ref_depth > 0,
            ref_max_depth=ref_depth,
            build_graph=True,
            output_obsidian=use_obsidian,
            output_index=True
        )

        agent = AutoAgent(self.llm_client, agent_config, output_topic=topic)
        result = agent.run(topic)

        self.console.print(f"\n[green]✅ 完成！[/green]")
        self.console.print(f"  分析论文: {result.get('papers_analyzed', 0)} 篇")
        self.console.print(f"  输出文件: {len(result.get('output_files', []))} 个")

        # 更新内部状态
        self.papers = agent.papers
        self.analyses = agent.analyses

    def run(self) -> None:
        """运行CLI主循环"""
        self.show_welcome()

        # 命令补全
        commands = ["auto", "search", "list", "analyze", "generate", "config", "help", "quit"]
        completer = WordCompleter(commands, ignore_case=True)

        while True:
            try:
                # 读取命令
                user_input = self.session.prompt(
                    HTML("<ansibright_blue>PaperReader></ansibright_blue> "),
                    completer=completer
                ).strip()

                if not user_input:
                    continue

                # 解析命令
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else None

                # 执行命令
                if cmd == "auto":
                    self.cmd_auto(arg)
                elif cmd == "search":
                    self.cmd_search(arg)
                elif cmd == "list":
                    self.cmd_list()
                elif cmd == "analyze" or cmd == "analyse":
                    self.cmd_analyze()
                elif cmd == "generate" or cmd == "gen":
                    self.cmd_generate()
                elif cmd == "config":
                    self.cmd_config()
                elif cmd == "help":
                    self.show_help()
                elif cmd == "quit" or cmd == "exit" or cmd == "q":
                    self.console.print("[cyan]再见！[/cyan]")
                    break
                else:
                    self.console.print(f"[red]未知命令: {cmd}[/red]")
                    self.console.print("输入 'help' 查看可用命令")

            except KeyboardInterrupt:
                self.console.print("\n[cyan]使用 'quit' 命令退出[/cyan]")
            except Exception as e:
                self.console.print(f"[red]错误: {e}[/red]")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="AI Paper Reader - 学术论文智能阅读助手")
    parser.add_argument(
        "-c", "--config",
        help="配置文件路径",
        default=None
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="研究主题（一键运行模式）"
    )
    parser.add_argument(
        "--papers",
        type=int,
        help="最大论文数量",
        default=20
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="不下载PDF"
    )
    parser.add_argument(
        "--expand-refs",
        action="store_true",
        help="启用参考文献扩充"
    )
    parser.add_argument(
        "--only-obsidian",
        action="store_true",
        help="只生成Obsidian知识库"
    )

    args = parser.parse_args()

    # 一键运行模式
    if args.topic:
        console = Console()
        console.print(f"[bold cyan]🚀 一键运行模式: {args.topic}[/bold cyan]")

        # 初始化LLM客户端
        config = get_config(args.config)
        llm_config = config.llm
        api_key = llm_config.get("api_key", "")

        if not api_key:
            console.print("[red]错误: 未配置LLM API密钥！[/red]")
            console.print("请设置 LLM_API_KEY 环境变量或在 .env 文件中配置")
            return

        try:
            llm_client = LLMClient(
                api_key=api_key,
                base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
                model=llm_config.get("model", "gpt-3.5-turbo"),
                temperature=llm_config.get("temperature", 0.3),
                max_tokens=llm_config.get("max_tokens", 4000)
            )

            # 创建Agent配置
            agent_config = AgentConfig(
                max_papers=args.papers,
                download_pdfs=not args.no_pdf,
                analyze_full_text=not args.no_pdf,
                expand_references=args.expand_refs,
                build_graph=True,
                output_obsidian=True,
                output_index=not args.only_obsidian
            )

            # 运行Agent
            agent = AutoAgent(llm_client, agent_config, output_topic=args.topic)
            result = agent.run(args.topic)

            console.print(f"\n[green]✅ 完成！共分析 {result.get('papers_analyzed', 0)} 篇论文[/green]")
            console.print(f"[cyan]输出文件:[/cyan]")
            for file in result.get('output_files', []):
                console.print(f"  - {file}")

        except Exception as e:
            console.print(f"[red]运行失败: {e}[/red]")
            import traceback
            traceback.print_exc()

    # 交互模式
    else:
        cli = PaperReaderCLI(config_path=args.config)
        cli.run()


if __name__ == "__main__":
    main()
