# Meta信息规范

[English](METADATA.md)

每条收录记录对应一个规范化网址，不等同于一个域名或公司。英文、中文简介放在同一记录中，生成完全相同的列表顺序。字段由 [JSON Schema 2020-12](schema/resource.schema.json) 定义；未声明的额外字段不能通过校验。

| 字段 | 含义 |
|---|---|
| `id` | 由规范网址的SHA-256生成稳定的`url-`标识。 |
| `name`、`kind` | 原始项目名称与主要资源类型；品牌、项目名称不强行翻译。 |
| `website`、`canonical_url` | 直接资源链接与去重身份；原始登记流程合并仓库分支/文件别名及追踪片段。 |
| `detail_url` | 代表记录在BotsTest的公开详情页。 |
| `summary.en`、`summary.zh-CN` | 先英文、后中文的简短用途说明，不宣称性能、兼容性或采购适用性。 |
| `category`、`subcategories` | 稳定分类slug与双语显示名称。 |
| `primary_subcategory` | 唯一展示分组，避免列表重复；其他子类保留在结构化数据中。无准确对应时为`null`。 |
| `classification` | 沿用分类或明确编辑修正，保存原始分类/子类及双语依据；沿用的能力标签不是独立认证。 |
| `relevance` | `direct`为直接具身/机器人资源，`supporting`为物理、RL、几何、控制等基础支持工具。 |
| `editorial` | 收录决定、复核日期、方法及双语原因；排除与待复核记录另行完整保留。 |
| `access` | 目录历史标签：开放、商业或研究；不是价格或法律使用权限。 |
| `licenses.repository` | 观测到的仓库许可标识、来源与日期，缺失时明确为`unknown`。 |
| `licenses.weights`、`.dataset`、`.hardware_design` | 权重、数据、硬件设计分别记录；未逐项核验的标识保持`null`，不复制仓库许可。 |
| `repository` | 公开仓库owner/name、主要语言、归档状态及最后观测推送时间；非仓库资源为`null`。 |
| `verification` | 历史网址状态、时间、方法；本版`new_check_performed=false`。 |
| `sources` | 公开发现来源与溯源入口，不代替项目官方文档。 |
| `tags`、`topics`、`applications` | 导入的目录发现标签，不证明功能支持或部署适用性。 |
| `aliases` | 共享同一网址的公开记录；不相加别名分数。 |
| `ranking` | 原有目录分数、原始/归一化PR、原始质量分量、代表节点、算法与时间。 |

未知与否、0不同。缺少许可不等于允许使用；归档不等于一定无法使用；网址可达也不等于安装测试通过。时间保留原时区偏移，编辑复核日期与来源观测日期独立。

## 文件职责

- `data/source-snapshot.json`：公开来源图及目录观测字段，用于复现历史分数。
- `data/url-registry.json`：1,000个历史核验候选网址。
- `data/reviews/*.json`：每个候选的一条双语编辑决定。
- `data/classification-overrides.json`：有依据的分类修正；不改原始来源或历史分数。
- `data/display-names.json`：发布者名称与大小写规范；原始名称保留在别名中。
- `data/resources.json`：符合schema的收录记录。
- `data/decisions.json`：全体收录、排除、待复核台账。
- `data/resources.csv`：扁平表格；空单元格表示缺失，不自动当0。
- `data/summary.json`、`data/checksums.json`：数量与SHA-256完整性清单。

修改原始/复核输入后运行生成与校验命令，不只手改一种语言。新来源观测必须附实际日期及方法；重新生成文件不刷新核验时间。
