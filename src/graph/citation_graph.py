"""
CitationGraph - 引用关系图模块
构建论文引用关系图，支持导出多种格式
"""

import json
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
from pathlib import Path


class CitationGraph:
    """
    论文引用关系图
    用于构建和分析论文之间的引用关系
    """

    def __init__(self):
        """初始化引用图"""
        # 节点：{paper_id: metadata}
        self.nodes: Dict[str, Dict[str, Any]] = {}

        # 边：{paper_id: [cited_paper_ids]}
        self.edges: Dict[str, List[str]] = defaultdict(list)

        # 反向边（被引用）：{paper_id: [citing_paper_ids]}
        self.reverse_edges: Dict[str, List[str]] = defaultdict(list)

    def add_paper(
        self,
        paper_id: str,
        title: str = "",
        authors: List[str] = None,
        year: int = None,
        arxiv_url: str = "",
        **kwargs
    ) -> None:
        """
        添加论文节点

        Args:
            paper_id: 论文ID（如arXiv ID）
            title: 论文标题
            authors: 作者列表
            year: 发表年份
            arxiv_url: arXiv链接
            **kwargs: 其他元数据
        """
        self.nodes[paper_id] = {
            "id": paper_id,
            "title": title,
            "authors": authors or [],
            "year": year,
            "arxiv_url": arxiv_url,
            **kwargs
        }

    def add_citation(self, from_id: str, to_id: str) -> None:
        """
        添加引用关系

        Args:
            from_id: 引用论文ID
            to_id: 被引用论文ID
        """
        if from_id not in self.edges:
            self.edges[from_id] = []
        if to_id not in self.edges:
            self.edges[to_id] = []

        # 避免重复
        if to_id not in self.edges[from_id]:
            self.edges[from_id].append(to_id)
            self.reverse_edges[to_id].append(from_id)

    def add_paper_from_analysis(self, analysis: Any, references: List[str] = None) -> None:
        """
        从PaperAnalysis对象添加论文

        Args:
            analysis: PaperAnalysis对象
            references: 参考文献ID列表
        """
        paper_id = analysis.arxiv_id or analysis.title

        self.add_paper(
            paper_id=paper_id,
            title=analysis.title,
            authors=analysis.authors.split(", ") if isinstance(analysis.authors, str) else analysis.authors,
            year=int(analysis.published[:4]) if analysis.published else None,
            arxiv_url=analysis.arxiv_url,
            category=analysis.category,
            summary=analysis.summary
        )

        # 添加引用关系
        if references:
            for ref_id in references:
                self.add_citation(paper_id, ref_id)

    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """
        获取论文信息

        Args:
            paper_id: 论文ID

        Returns:
            论文元数据或None
        """
        return self.nodes.get(paper_id)

    def get_citations(self, paper_id: str) -> List[str]:
        """
        获取论文引用的其他论文

        Args:
            paper_id: 论文ID

        Returns:
            被引用论文ID列表
        """
        return self.edges.get(paper_id, [])

    def get_cited_by(self, paper_id: str) -> List[str]:
        """
        获取引用该论文的其他论文

        Args:
            paper_id: 论文ID

        Returns:
            引用论文ID列表
        """
        return self.reverse_edges.get(paper_id, [])

    def find_related(self, paper_id: str, max_depth: int = 2) -> Set[str]:
        """
        查找相关论文（引用+被引用）

        Args:
            paper_id: 论文ID
            max_depth: 最大深度

        Returns:
            相关论文ID集合
        """
        related = set()
        visited = set()
        queue = [(paper_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)

            # 添加引用的论文
            for cited_id in self.get_citations(current_id):
                if cited_id not in visited:
                    related.add(cited_id)
                    queue.append((cited_id, depth + 1))

            # 添加被引用的论文
            for citing_id in self.get_cited_by(current_id):
                if citing_id not in visited:
                    related.add(citing_id)
                    queue.append((citing_id, depth + 1))

        return related

    def find_shortest_path(self, from_id: str, to_id: str) -> Optional[List[str]]:
        """
        查找两篇论文之间的最短路径

        Args:
            from_id: 起始论文ID
            to_id: 目标论文ID

        Returns:
            路径ID列表，如果不连通返回None
        """
        from collections import deque

        queue = deque([[from_id]])
        visited = {from_id}

        while queue:
            path = queue.popleft()
            current_id = path[-1]

            if current_id == to_id:
                return path

            # 搜索相邻节点
            neighbors = self.get_citations(current_id) + self.get_cited_by(current_id)

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    queue.append(new_path)

        return None

    def get_clusters(self, min_size: int = 3) -> List[Set[str]]:
        """
        获取论文簇（弱连通分量）

        Args:
            min_size: 最小簇大小

        Returns:
            论文簇列表
        """
        visited = set()
        clusters = []

        for paper_id in self.nodes:
            if paper_id in visited:
                continue

            # BFS遍历连通分量
            cluster = set()
            queue = [paper_id]

            while queue:
                current_id = queue.pop(0)

                if current_id in visited:
                    continue

                visited.add(current_id)
                cluster.add(current_id)

                # 添加相邻节点
                neighbors = self.get_citations(current_id) + self.get_cited_by(current_id)
                queue.extend(n for n in neighbors if n not in visited)

            if len(cluster) >= min_size:
                clusters.append(cluster)

        return clusters

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取图统计信息

        Returns:
            统计信息字典
        """
        num_nodes = len(self.nodes)
        num_edges = sum(len(edges) for edges in self.edges.values())

        # 计算度数
        in_degrees = [len(self.get_cited_by(pid)) for pid in self.nodes]
        out_degrees = [len(self.get_citations(pid)) for pid in self.nodes]

        return {
            "num_papers": num_nodes,
            "num_citations": num_edges,
            "avg_in_degree": sum(in_degrees) / num_nodes if num_nodes > 0 else 0,
            "avg_out_degree": sum(out_degrees) / num_nodes if num_nodes > 0 else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0
        }

    def export_json(self, file_path: str) -> None:
        """
        导出为JSON格式

        Args:
            file_path: 输出文件路径
        """
        data = {
            "nodes": [
                {**node, "id": pid}
                for pid, node in self.nodes.items()
            ],
            "edges": [
                {"from": from_id, "to": to_id}
                for from_id, to_ids in self.edges.items()
                for to_id in to_ids
            ]
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def export_graphml(self, file_path: str) -> None:
        """
        导出为GraphML格式（可用Gephi打开）

        Args:
            file_path: 输出文件路径
        """
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"')
        lines.append('    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
        lines.append('    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns')
        lines.append('    http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">')

        # 定义键
        lines.append('  <key id="title" for="node" attr.name="title" attr.type="string"/>')
        lines.append('  <key id="year" for="node" attr.name="year" attr.type="int"/>')
        lines.append('  <key id="category" for="node" attr.name="category" attr.type="string"/>')

        # 图开始
        lines.append('  <graph id="citation_graph" edgedefault="directed">')

        # 节点
        for pid, node in self.nodes.items():
            lines.append(f'    <node id="{pid}">')
            lines.append(f'      <data key="title">{node.get("title", "")}</data>')
            if node.get("year"):
                lines.append(f'      <data key="year">{node["year"]}</data>')
            if node.get("category"):
                lines.append(f'      <data key="category">{node["category"]}</data>')
            lines.append('    </node>')

        # 边
        edge_id = 0
        for from_id, to_ids in self.edges.items():
            for to_id in to_ids:
                lines.append(f'    <edge id="e{edge_id}" source="{from_id}" target="{to_id}"/>')
                edge_id += 1

        # 图结束
        lines.append('  </graph>')
        lines.append('</graphml>')

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def export_obsidian_links(self) -> Dict[str, List[str]]:
        """
        生成Obsidian格式的链接关系

        Returns:
            {paper_id: [linked_paper_ids]} 字典
        """
        links = {}

        for paper_id in self.nodes:
            related = self.find_related(paper_id, max_depth=1)
            links[paper_id] = list(related)

        return links


def build_graph_from_analyses(analyses: List[Any]) -> CitationGraph:
    """
    从分析结果列表构建引用图

    Args:
        analyses: PaperAnalysis对象列表

    Returns:
        CitationGraph对象
    """
    graph = CitationGraph()

    for analysis in analyses:
        graph.add_paper_from_analysis(analysis)

    return graph
