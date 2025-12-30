"""
AutoAgent - 自动化流程Agent
一键式完成：搜索→下载→分析→扩充→图谱→输出
"""

import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from .query_expander import QueryExpander
from .reference_miner import ReferenceMiner
from ..apis.llm_client import LLMClient
from ..apis.arxiv_api import ArxivAPI, ArxivPaper
from ..apis.scihub_api import SciHubAPI, SciHubPaper
from ..apis.paper_base import BasePaper, BasePaperAPI, PaperSource, get_paper_source
from ..analyzers.paper_analyzer import PaperAnalyzer, PaperAnalysis
from ..graph.citation_graph import CitationGraph
from ..graph.obsidian_renderer import ObsidianRenderer
from ..utils.config import get_config
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn


@dataclass
class AgentConfig:
    """Agent配置"""
    max_papers: int = 20
    download_pdfs: bool = True
    analyze_full_text: bool = True
    expand_references: bool = False
    ref_max_depth: int = 1
    build_graph: bool = True
    output_obsidian: bool = True
    output_index: bool = True
    paper_sources: List[str] = None  # 论文数据源列表 ['arxiv', 'scihub']

    def __post_init__(self):
        if self.paper_sources is None:
            self.paper_sources = ['arxiv']  # 默认只使用arxiv


class AutoAgent:
    """
    自动化论文处理Agent
    一键式完成从搜索到输出的完整流程
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: AgentConfig = None,
        output_topic: str = None,
        search_field: str = "all"
    ):
        """
        初始化AutoAgent

        Args:
            llm_client: LLM客户端
            config: Agent配置
            output_topic: 输出主题（用于创建关键词特定的输出文件夹）
            search_field: 搜索字段（all/ti/abs）
        """
        self.llm_client = llm_client
        self.config = config or AgentConfig()
        self.output_topic = output_topic or "default"
        self.search_field = search_field
        self.console = Console()

        # 初始化各个组件
        self.query_expander = QueryExpander(llm_client)
        self.reference_miner = None  # 将在初始化paper_apis后设置

        # 初始化多个论文数据源API
        self.paper_apis: Dict[str, BasePaperAPI] = {}
        self._init_paper_apis()

        # 初始化分析器
        self.analyzer = PaperAnalyzer(
            llm_client=llm_client,
            analyze_full_text=self.config.analyze_full_text
        )

        # 结果存储
        self.papers: List[BasePaper] = []
        self.analyses: List[PaperAnalysis] = []
        self.graph: Optional[CitationGraph] = None

    def _init_paper_apis(self):
        """初始化论文数据源API"""
        config = get_config()

        # 初始化arXiv API
        if 'arxiv' in self.config.paper_sources:
            search_config = config.search
            self.paper_apis['arxiv'] = ArxivAPI(
                max_results=self.config.max_papers,
                sort_by=search_config.get("sort_by", "relevance"),
                categories=search_config.get("categories", []),
                search_field=self.search_field
            )

        # 初始化Sci-Hub API
        if 'scihub' in self.config.paper_sources:
            scihub_config = config.get("scihub", {})
            if scihub_config.get("enabled", True):
                self.paper_apis['scihub'] = SciHubAPI(
                    base_url=scihub_config.get("base_url", ""),
                    timeout=scihub_config.get("timeout", 60),
                    max_retries=scihub_config.get("max_retries", 3)
                )

        # 初始化ReferenceMiner（使用arXiv API）
        if 'arxiv' in self.paper_apis:
            self.reference_miner = ReferenceMiner(self.paper_apis['arxiv'])

    def run(self, topic: str) -> Dict[str, Any]:
        """
        执行完整的自动化流程

        Args:
            topic: 研究主题

        Returns:
            执行结果字典
        """
        self.console.print(Panel.fit(
            f"[bold cyan]🚀 开始自动化处理: {topic}[/bold cyan]"
        ))

        result = {
            "topic": topic,
            "start_time": datetime.now().isoformat(),
            "papers_found": 0,
            "papers_analyzed": 0,
            "output_files": []
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:

            # 步骤1-2: AI拓展搜索词并搜索arXiv论文（带重试机制）
            max_retries = 5
            retry_count = 0
            self.papers = []
            failed_keywords_history = []  # 记录失败的关键词

            while retry_count < max_retries and not self.papers:
                retry_count += 1

                # 步骤1: AI拓展搜索词
                task = progress.add_task(f"[cyan]AI拓展搜索词 (尝试 {retry_count}/{max_retries})...", total=1)

                # 首次使用原始主题，后续传入失败的关键词历史
                if retry_count == 1:
                    search_topic = topic
                    expanded_queries = self.query_expander.expand_query(search_topic)
                else:
                    # 将失败的关键词历史传递给LLM
                    expanded_queries = self.query_expander.expand_query_with_feedback(
                        topic,
                        failed_keywords_history
                    )

                progress.update(task, completed=1)
                self.console.print(f"[green]✓[/green] 生成 {len(expanded_queries)} 个搜索词")

                # 步骤2: 搜索arXiv论文
                task = progress.add_task(f"[cyan]搜索arXiv论文 (尝试 {retry_count}/{max_retries})...", total=len(expanded_queries))
                self.papers = self._search_papers(expanded_queries, progress, task)

                if self.papers:
                    result["papers_found"] = len(self.papers)
                    self.console.print(f"[green]✓[/green] 找到 {len(self.papers)} 篇论文")
                elif retry_count < max_retries:
                    # 记录失败的关键词
                    failed_keywords_history.extend(expanded_queries)
                    self.console.print(f"[yellow]未找到相关论文，正在重新生成关键词... ({retry_count}/{max_retries})[/yellow]")

            if not self.papers:
                self.console.print(f"[red]已尝试 {max_retries} 次，仍未找到相关论文[/red]")
                self.console.print(f"[dim]使用过的关键词: {failed_keywords_history}[/dim]")
                self.console.print("[yellow]建议：尝试更换更通用的搜索词或检查主题是否过于偏门[/yellow]")
                return result

            # 步骤3: 下载PDF
            pdf_paths = {}
            if self.config.download_pdfs:
                task = progress.add_task("[cyan]下载PDF...", total=len(self.papers))
                pdf_paths = self._download_pdfs(self.papers, progress, task)

            # 步骤4: 分析论文
            task = progress.add_task("[cyan]分析论文...", total=len(self.papers))
            self.analyses = self._analyze_papers(self.papers, pdf_paths, progress, task)
            result["papers_analyzed"] = len(self.analyses)
            self.console.print(f"[green]✓[/green] 分析完成 {len(self.analyses)} 篇")

            # 步骤5: 构建知识图谱
            if self.config.build_graph:
                task = progress.add_task("[cyan]构建知识图谱...", total=1)
                self.graph = self._build_graph(self.analyses)
                progress.update(task, completed=1)
                self.console.print(f"[green]✓[/green] 知识图谱构建完成")

            # 步骤6: 参考文献扩充（支持Ctrl+C中断）
            if self.config.expand_references:
                # 估算总任务数（每层论文数 × 层数）
                estimated_total = len(self.papers) * self.config.ref_max_depth
                task = progress.add_task("[cyan]挖掘参考文献...", total=estimated_total)

                try:
                    expanded_papers = self._expand_references(self.papers, progress, task)
                    if expanded_papers:
                        self.papers.extend(expanded_papers)
                        self.console.print(f"[green]✓[/green] 扩充 {len(expanded_papers)} 篇相关论文")
                except KeyboardInterrupt:
                    # 用户中断，继续后续步骤
                    self.console.print("[yellow]参考文献挖掘已中断，继续后续步骤...[/yellow]")
                    progress.update(task, completed=estimated_total)

            # 步骤7: 生成输出
            task = progress.add_task("[cyan]生成输出...", total=3)
            output_files = self._generate_outputs(topic)
            result["output_files"] = output_files
            progress.update(task, completed=3)

        result["end_time"] = datetime.now().isoformat()

        self.console.print(Panel.fit(
            f"[bold green]✅ 处理完成！[/bold green]\n"
            f"论文数: {len(self.analyses)}\n"
            f"输出文件: {len(output_files)} 个"
        ))

        return result

    def _search_papers(
        self,
        queries: List[str],
        progress,
        task: int
    ) -> List[BasePaper]:
        """
        使用多个搜索词搜索论文（支持多数据源）

        Args:
            queries: 搜索词列表
            progress: Progress对象
            task: 任务ID

        Returns:
            论文列表（去重）
        """
        all_papers = []
        seen_ids = set()

        # 计算每个数据源应该获取的论文数
        num_sources = len(self.paper_apis)
        if num_sources == 0:
            return []

        papers_per_source = max(5, self.config.max_papers // num_sources)

        # 从每个数据源搜索
        for source_name, api in self.paper_apis.items():
            for query in queries:
                try:
                    # 根据数据源类型调整搜索参数
                    if source_name == 'scihub':
                        # Sci-Hub主要支持DOI查询，对于普通查询使用CrossRef
                        papers = api.search(
                            query=query,
                            max_results=1,
                            use_crossref=True
                        )
                    else:
                        # arXiv等支持普通搜索
                        papers = api.search(
                            query=query,
                            max_results=papers_per_source
                        )

                    for paper in papers:
                        # 使用paper_id作为唯一标识
                        paper_id = paper.paper_id
                        if paper_id and paper_id not in seen_ids:
                            seen_ids.add(paper_id)
                            all_papers.append(paper)

                            if len(all_papers) >= self.config.max_papers:
                                break

                except Exception as e:
                    self.console.print(f"[yellow]{source_name}搜索失败 ({query}): {e}[/yellow]")

                progress.update(task, advance=1)

                if len(all_papers) >= self.config.max_papers:
                    break

            if len(all_papers) >= self.config.max_papers:
                break

        return all_papers[:self.config.max_papers]

    def _download_pdfs(
        self,
        papers: List[BasePaper],
        progress,
        task: int
    ) -> Dict[str, str]:
        """
        下载论文PDF（支持多数据源）

        Args:
            papers: 论文列表
            progress: Progress对象
            task: 任务ID

        Returns:
            {paper_id: pdf_path} 字典
        """
        pdf_paths = {}
        papers_dir = Path(get_config().output.get("papers_dir", "./data/papers"))
        papers_dir.mkdir(parents=True, exist_ok=True)

        for paper in papers:
            try:
                from ..utils.io import safe_filename_from_title

                # 使用paper_id作为文件名前缀，如果没有则用标题的hash
                paper_id = paper.paper_id or hashlib.md5(paper.title.encode()).hexdigest()[:8]
                filename = f"{paper_id}_{safe_filename_from_title(paper.title)}.pdf"
                pdf_path = papers_dir / filename

                if not pdf_path.exists():
                    # 根据论文来源选择对应的API下载
                    paper_source = get_paper_source(paper)
                    if paper_source == PaperSource.SCIHUB and 'scihub' in self.paper_apis:
                        self.paper_apis['scihub'].download_pdf(paper, str(pdf_path))
                    elif paper_source == PaperSource.ARXIV and 'arxiv' in self.paper_apis:
                        self.paper_apis['arxiv'].download_pdf(paper, str(pdf_path))
                    else:
                        # 尝试使用通用下载方法
                        if paper.pdf_url:
                            self._download_pdf_direct(paper.pdf_url, str(pdf_path))

                pdf_paths[paper_id] = str(pdf_path)

            except Exception as e:
                self.console.print(f"[yellow]下载失败 ({paper.title[:30]}...): {e}[/yellow]")

            progress.update(task, advance=1)

        return pdf_paths

    def _download_pdf_direct(self, pdf_url: str, save_path: str, timeout: int = 120) -> bool:
        """
        直接下载PDF

        Args:
            pdf_url: PDF链接
            save_path: 保存路径
            timeout: 超时时间

        Returns:
            是否成功
        """
        try:
            import requests
            response = requests.get(pdf_url, timeout=timeout, stream=True)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception:
            return False

    def _analyze_papers(
        self,
        papers: List[BasePaper],
        pdf_paths: Dict[str, str],
        progress,
        task: int
    ) -> List[PaperAnalysis]:
        """
        分析论文

        Args:
            papers: 论文列表
            pdf_paths: PDF路径字典
            progress: Progress对象
            task: 任务ID

        Returns:
            分析结果列表
        """
        analyses = []

        for paper in papers:
            try:
                paper_id = paper.paper_id or hashlib.md5(paper.title.encode()).hexdigest()[:8]
                pdf_path = pdf_paths.get(paper_id)

                # 根据论文类型调用相应的分析方法
                paper_source = get_paper_source(paper)
                if paper_source == PaperSource.ARXIV:
                    analysis = self.analyzer.analyze_from_arxiv(paper, pdf_path)
                elif isinstance(paper, BasePaper):
                    # 通用分析方法
                    analysis = self.analyzer.analyze_from_paper(paper, pdf_path)
                else:
                    # 兜底：使用通用分析方法
                    analysis = self.analyzer.analyze_from_paper(paper, pdf_path)

                analyses.append(analysis)
            except Exception as e:
                self.console.print(f"[yellow]分析失败 ({paper.title[:30]}...): {e}[/yellow]")

            progress.update(task, advance=1)

        return analyses

    def _build_graph(self, analyses: List[PaperAnalysis]) -> CitationGraph:
        """
        构建引用关系图

        Args:
            analyses: 分析结果列表

        Returns:
            CitationGraph对象
        """
        graph = CitationGraph()

        for analysis in analyses:
            graph.add_paper_from_analysis(analysis)

        return graph

    def _expand_references(
        self,
        papers: List[BasePaper],
        progress,
        task: int
    ) -> List[BasePaper]:
        """
        基于参考文献递归扩充论文
        优先使用LLM分析时提取的参考文献
        支持多数据源：arXiv + Sci-Hub

        Args:
            papers: 当前论文列表
            progress: Progress对象
            task: 任务ID

        Returns:
            新发现的论文列表
        """
        expanded = []
        seen_ids = set(p.paper_id for p in papers if p.paper_id)
        papers_dir = Path(get_config().output.get("papers_dir", "./data/papers"))
        max_depth = self.config.ref_max_depth
        total_found = 0

        # 构建论文分析结果索引（paper_id -> analysis）
        analysis_map = {}
        for analysis in self.analyses:
            paper_id = analysis.paper_id or analysis.arxiv_id
            if paper_id:
                analysis_map[paper_id] = analysis

        # 递归挖掘参考文献
        def mine_recursive(current_papers: List[BasePaper], current_depth: int) -> None:
            nonlocal total_found
            if current_depth > max_depth:
                return

            depth_papers_found = 0
            total_to_process = len(current_papers)

            for i, paper in enumerate(current_papers):
                # 更新进度显示
                progress.update(
                    task,
                    description=f"[cyan]挖掘参考文献 (深度{current_depth}/{max_depth}, {i+1}/{total_to_process}, 已发现{total_found}篇)..."
                )

                try:
                    found_papers = {}  # {title: paper_obj}
                    paper_id = paper.paper_id or hashlib.md5(paper.title.encode()).hexdigest()[:8]

                    # 优先使用LLM分析时提取的参考文献
                    reference_titles = []
                    if paper_id in analysis_map:
                        llm_refs = analysis_map[paper_id].references
                        if llm_refs:
                            self.console.print(f"[dim]  使用LLM提取的 {len(llm_refs)} 篇参考文献[/dim]")
                            reference_titles = llm_refs

                    # 如果LLM提取失败或为空，回退到PDF提取
                    if not reference_titles and paper_id:
                        pdf_files = list(papers_dir.glob(f"{paper_id}_*.pdf"))
                        if pdf_files and 'arxiv' in self.paper_apis:
                            from .reference_miner import ReferenceMiner
                            miner = ReferenceMiner(self.paper_apis['arxiv'])
                            references = miner.extract_references(str(pdf_files[0]))
                            # 从Reference对象中提取标题
                            reference_titles = [ref.title for ref in references if ref.title]

                    # 多数据源搜索参考文献
                    for title in reference_titles[:30]:  # 限制处理数量
                        # 1. 先尝试arXiv
                        if 'arxiv' in self.paper_apis:
                            try:
                                results = self.paper_apis['arxiv'].search(
                                    query=f"ti:{title}",
                                    max_results=1,
                                    timeout=3
                                )
                                if results:
                                    new_paper = results[0]
                                    if new_paper.paper_id not in seen_ids:
                                        found_papers[title] = new_paper
                                        seen_ids.add(new_paper.paper_id)
                                        continue
                            except Exception:
                                pass

                        # 2. 如果arXiv没找到，尝试Sci-Hub/CrossRef
                        if title not in found_papers and 'scihub' in self.paper_apis:
                            try:
                                results = self.paper_apis['scihub'].search(
                                    query=title,
                                    max_results=1,
                                    use_crossref=True
                                )
                                if results:
                                    new_paper = results[0]
                                    if new_paper.paper_id and new_paper.paper_id not in seen_ids:
                                        found_papers[title] = new_paper
                                        seen_ids.add(new_paper.paper_id)
                            except Exception:
                                pass

                    # 添加新发现的论文
                    for title, new_paper in found_papers.items():
                        expanded.append(new_paper)
                        total_found += 1
                        depth_papers_found += 1

                        # 显示来源
                        source = get_paper_source(new_paper)
                        self.console.print(
                            f"[dim]  + [{current_depth}][{source.upper()}] {new_paper.title[:50]}...[/dim]"
                        )

                except KeyboardInterrupt:
                    # 用户中断，返回已发现的论文
                    self.console.print(f"\n[yellow]用户中断参考文献挖掘[/yellow]")
                    raise
                except Exception as e:
                    self.console.print(f"[dim]  挖掘失败 ({paper.title[:30]}...): {e}[/dim]")

                progress.update(task, advance=1)

                # 限制单层扩充数量避免过度膨胀
                if depth_papers_found >= 30:
                    break

            # 如果发现了新论文且未达到最大深度，继续递归
            if depth_papers_found > 0 and current_depth < max_depth:
                new_papers = expanded[-depth_papers_found:]
                mine_recursive(new_papers, current_depth + 1)

        try:
            mine_recursive(papers, 1)
        except KeyboardInterrupt:
            # 用户中断，继续后续步骤
            self.console.print(f"[yellow]参考文献挖掘已停止，共发现 {total_found} 篇新论文[/yellow]")

        return expanded

    def _generate_outputs(self, topic: str) -> List[str]:
        """
        生成所有输出文件

        Args:
            topic: 研究主题

        Returns:
            生成的文件路径列表
        """
        output_files = []
        config = get_config()

        # 创建关键词特定的输出文件夹
        from ..utils.io import sanitize_filename
        safe_topic = sanitize_filename(topic)
        topic_output_dir = Path("./output") / safe_topic
        topic_output_dir.mkdir(parents=True, exist_ok=True)

        # Obsidian知识库
        if self.config.output_obsidian:
            vault_path = topic_output_dir / "vault"
            renderer = ObsidianRenderer(str(vault_path))

            # 导出所有论文笔记
            paper_files = renderer.export_all_papers(self.analyses)
            output_files.extend(paper_files)

            # 生成主题索引
            topic_file = renderer.save_topic_index(
                topic,
                self.analyses,
                description=f"关于{topic}的研究论文集合"
            )
            output_files.append(topic_file)

            # 生成MOC
            moc_file = renderer.save_moc(self.analyses, self.graph)
            output_files.append(moc_file)

        # 索引文件
        if self.config.output_index:
            from ..renderers.markdown_gen import MarkdownGenerator

            generator = MarkdownGenerator()
            index_file = topic_output_dir / "papers_index.md"
            content = generator.generate_index(self.analyses, title=f"{topic} - 论文索引")

            index_file.parent.mkdir(parents=True, exist_ok=True)
            with open(index_file, "w", encoding="utf-8") as f:
                f.write(content)

            output_files.append(str(index_file))

        # 导出图谱
        if self.graph:
            graph_dir = topic_output_dir / "graph"
            graph_dir.mkdir(parents=True, exist_ok=True)

            json_file = graph_dir / "citation_graph.json"
            self.graph.export_json(str(json_file))
            output_files.append(str(json_file))

            graphml_file = graph_dir / "citation_graph.graphml"
            self.graph.export_graphml(str(graphml_file))
            output_files.append(str(graphml_file))

        return output_files


def run_auto_agent(topic: str, llm_client: LLMClient, **kwargs) -> Dict[str, Any]:
    """
    便捷函数：运行AutoAgent

    Args:
        topic: 研究主题
        llm_client: LLM客户端
        **kwargs: Agent配置参数

    Returns:
        执行结果字典
    """
    config = AgentConfig(**kwargs)
    agent = AutoAgent(llm_client, config, output_topic=topic)
    return agent.run(topic)
