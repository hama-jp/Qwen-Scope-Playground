# feat の発掘メモ — corpus.db 直接探索版

`feature_explorer.py` で corpus.db (3 ソース 999 サンプル, layer 10) を多軸スキャン → register/BOS feature を除外した 27419 件から、tour に追加する価値がある **特に面白い 12 個** を厳選。

各エントリは `feat # / 概念ラベル / source 分布 / 上位 3 snippet (peak token を【】で囲む) / 解釈` の順。

---

## A. 「これだけ?」と驚く程度に narrow な Python 専用 feature

### feat #2371 — `from __future__ import` の `future`

| | |
| --- | --- |
| source | EN=0, CODE=20, JA=0 |
| mean act | 65.4 (cv=0.34) |

snippet:
- `# -*- coding: utf-8 -*-\nfrom __【future】__ import absolute_import, print_function, unicode_literals`
- `# -*- coding: utf-8 -*-\nfrom __【future】__ import unicode_literals\nimport os\n`

解釈: Python 2/3 互換性のための `__future__` モジュール、その「future」という単一トークンに専属する feature。SAE は **「Python の特殊な互換 import」** という 1 行のパターンを 1 つの番号に切り出している。`import` や `from` 自体は当然別 feature が担当。

### feat #28927 — `for X in Y:` の `for`

| | |
| --- | --- |
| source | EN=0, CODE=20, JA=0 |
| mean act | 50.3 (cv=0.02 — 極めて低分散) |

snippet:
- `params = {}\n   【 for】 key in args:\n        val = args[key]`
- `       【 for】 name in self.cores:\n            pr_info(`
- `   【 for】 device in devices_set:\n        found_2min = False`

解釈: cv=0.02 は本ハンズオンで観測した中でもトップクラスの sharp detector。**Python の `for` キーワード、しかも左にコロン以外のもの(改行・空白)が来るループ開始位置** にだけ反応する。

### feat #1419 — `for X in Y:` の `in` (#28927 のペア feature)

| | |
| --- | --- |
| source | EN=0, CODE=20, JA=0 |
| mean act | 45.9 (cv=0.02) |

snippet:
- `for line【 in】 output.splitlines():\n		if not line:`
- `for event【 in】 pygame.event.get():`
- `for l【 in】 f:\n                yield l.split()`

解釈: feat 28927 と一緒に物語る。SAE は `for ... in ...:` 構文を **for 用 feature と in 用 feature の 2 つに分けて** 表現している。同じ `in` でも `if x in list` 等の `in` 演算子用は別 feature(おそらく)になっているはず。

### feat #8087 — `len(X)` の関数名

| | |
| --- | --- |
| source | EN=0, CODE=20, JA=0 |
| mean act | 32.9 (cv=0.02) |

snippet:
- `if __name__ == '__main__':\n    if【 len】(sys.argv) == 2:`
- `def __【len】__(self):\n        return self.t.rnum()`
- `if【 len】(disconnected_controllers) == 0:`

解釈: Python ビルトインの `len` 専用。`__len__` のような dunder 形でも発火する。「**ライブラリ名でも普通の変数名でもなく、特定のビルトイン関数名 1 つに専属する feature**」が SAE の中に存在する。

---

## B. ライセンス文の「定型句」を捉える feature たち

### feat #29608 — GPL 「GNU」

| | |
| --- | --- |
| source | EN=0, CODE=20, JA=0 |
| mean act | 76.0 (cv=0.03) |

snippet:
- `# it under the terms of the【 GNU】 General Public License as published by\n# the Free`
- `# See the\n#【 GNU】 General Public License for more details.`

### feat #198 — GPL 「License」

| | |
| --- | --- |
| source | EN=0, CODE=20, JA=0 |
| mean act | 71.6 (cv=0.12) |

snippet:
- `# GNU General Public【 License】 for more details.\n\n# You should have received`

### feat #9750 — GPL 「(at your option)」

| | |
| --- | --- |
| source | EN=0, CODE=20, JA=0 |
| mean act | 70.4 (cv=0.26) |

snippet:
- `# of the License, or\n# (at your【 option】) any later version.`
- `## of the License, or\n## (at your【 option】) any later version.`

解釈: **GPL ライセンス preamble の特定単語を 3 つの feature が分担して捉えている**。`GNU` / `License` / `option` がそれぞれ別の番号で、ファイル冒頭のあのお馴染みのコメントブロックが SAE 上では複数 feature の同時発火として表現される。「ライセンス全般」という 1 つの feature ではなく、定型句のキー単語ごとに 1 個ずつ。

### feat #6460 — 同じ `the` でもライセンス文脈の `the`

| | |
| --- | --- |
| source | EN=0, CODE=20, JA=0 |
| mean act | 52.0 (cv=0.01) |

snippet:
- `to deal\nin【 the】 Software without restriction, including without limitation the rights`
- `the Free Software Foundation, either version 3 of【 the】 License, or\n# (at your option)`
- `# You should have received a copy of【 the】 GNU General Public License`

解釈: cv=0.01 は最低分散級の sharp detector。`the` という英語で最も頻出の語の中でも、**ライセンス boilerplate 文中の `the`** にだけ専属する feature。同じ `the` でも別の用法は別 feature が拾うはず。SAE が「**意味より文脈・register**」で feature を切るときの典型例。

---

## C. 日本語の機能語を 1 文字単位で捉える feature

### feat #7690 — 助詞「を」

| | |
| --- | --- |
| source | EN=0, CODE=0, JA=20 |
| mean act | 43.8 (cv=0.01) |

snippet:
- `ミニパラダイムは逆に混迷【を】深めた。`
- `アミルタイオスが反乱【を】起こし、紀元前404年`
- `産業地理学（農業【を】扱う農業地理学、工業を`

### feat #20262 — 助詞「に」

| | |
| --- | --- |
| source | EN=0, CODE=0, JA=20 |
| mean act | 42.2 (cv=0.01) |

snippet:
- `消えてしまった地形【に】適合していたにすぎない`
- `生物は水の特殊な物性【に】多くの事を依存しており`
- `ラッツェルが自然地理の条件【に】人類は強く影響を受ける`

### feat #17402 — 読点「、」

| | |
| --- | --- |
| source | EN=0, CODE=0, JA=20 |
| mean act | 21.4 (cv=0.02) |

snippet:
- `アンパサンド（&, ）は【、】並立助詞「…と…」を`
- `地理学は【、】大きく系統地理学と地誌学に`
- `用語「ヨーロッパ」は【、】歴史が展開する中で使`

解釈: 助詞「を」「に」、句読点「、」がそれぞれ独立した feature。**SAE は日本語の文法粒子をそれぞれ別の番号に切り出している**。英語の `the` のような頻出機能語と違って、日本語の助詞は意味的役割(目的格・場所格・列挙)が異なるので、それぞれ別 feature が割り振られているのが直感とも合う。tour §5 (Concept search) で見た「日本語の助詞 feature」は実体として **これらの個別の助詞 feature の集合** だと分かる。

---

## D. 概念が言語を超えて 1 feature にまとまる cross-lingual の例

### feat #25342 — 「国際 / international」

| | |
| --- | --- |
| source | EN=8, CODE=0, JA=12 |
| mean act | 44.9 (cv=0.04) |

snippet:
- `[JA] 14区の外れにあり、国【際】大学都市に面している。`
- `[JA] 国【際】地理学連合（IGU）`
- `[EN] (Only for【 international】 patients seeking treatment in India)`

解釈: 日本語の漢字「際」(=「国際」の 2 字目) と英語の「international」が **同じ feature 番号** で発火する。tour §3 で feat 16504(自由 / freedom)を観察したのと同じ「概念は言語に独立」の追加実証。**翻訳的な意味理解の片鱗** が SAE の中で観測できる。

### feat #14120 — 「経済 / economic」

| | |
| --- | --- |
| source | EN=14, CODE=0, JA=6 |
| mean act | 46.9 (cv=0.04) |

snippet:
- `[EN] balancing【 economic】 growth and ecological protection`
- `[EN] comprehensive global assessment of【 economic】 aspects of these issues.`
- `[JA] サーダートは、経【済】の開放などに舵を切る`

解釈: 同上、英語 ` economic` と日本語「経」+「済」(の 2 字目) が同 feature。専門語領域でも cross-lingual 対応が成立。

### feat #4957 — 「University / 引用文中の University」

| | |
| --- | --- |
| source | EN=14, CODE=0, JA=6 |
| mean act | 46.6 (cv=0.05) |

snippet:
- `[EN] Springfield Museum of Fine Arts\n【University】 of Utah, UT`
- `[JA] in resource management and conservation. Chicago, Ill.:【 University】 of Chicago Press.`
- `[EN] University【 of】 California Press: 1995.`

解釈: ` University` という英単語自体だが、**EN プロンプトでも JA プロンプト(日本語 Wikipedia の参考文献欄)でも発火する**。日本語コーパスの中でも引用書誌は英語で書くことが多いので、cross-source feature として現れる。「概念」というより「**書誌情報の記法**」を捉えている。

---

## E. その他で印象に残ったもの

### feat #5657 — 数字内の桁(小数点を含む)

| | |
| --- | --- |
| source | EN=7, CODE=9, JA=4 |
| mean act | 38.7 (cv=0.09) |

snippet:
- `[JA] 二酸化炭素が0.04【1】%という構成`
- `[JA] 世帯あたり1.7【5】人で、国平均の1.86`
- `[CODE] 0 0.693【1】471806 1.0`

解釈: 数値リテラルの **小数点以下の特定の桁** に発火。同様に feat #14321 が **西暦 4 桁の中央桁**(`200【0】8`, `19【9】9`)に専属。SAE は **数字の塊** ではなく **桁ごとの位置** で別 feature を持っている。tour §5 「数字の列挙」で見た 5 件分業の延長。

### feat #18310 — 「イスラム」「イスラム教」のム

| | |
| --- | --- |
| source | EN=7, CODE=0, JA=13 |
| mean act | 48.0 (cv=0.09) |

snippet:
- `アラブ世界、イスラ【ム】世界の一国である。`
- `主に信仰されるイスラ【ム】教がある。`
- `イスラ【ム】王朝 \n7世紀にイスラム化`

解釈: 「イスラム」というカタカナ語を **「ラ」の次の「ム」一字** にトークナイザが切り、その「ム」に発火する feature。**Qwen のトークナイザが日本語のカタカナ語をどう切るかが SAE の feature 構造に影響している** 一例。「イスラム」全体を 1 feature にすると思いきや、トークン境界に従って分業している。

### feat #14646 — 「Alzheimer's」

| | |
| --- | --- |
| source | EN=20, CODE=0, JA=0 |
| mean act | 50.6 (cv=0.38) |

snippet:
- `Al【zheimer】's is a progressive neurodegenerative disease`
- `another classic hallmark of【 Alzheimer】's, neurofibrillary tangle in nerve`
- `human trials testing this type of【 Alzheimer】's vaccine`

解釈: **特定の固有名詞専用 feature**。SAE は「アルツハイマー病」という疾患名を 1 つの番号に持っている。corpus に同じ記事が複数チャンクで入っているので 20 件全部 EN なのは偏りで、もし日本語コーパスに「アルツハイマー」を含む文があれば cross-lingual で出てくる可能性が高い(将来の検証ネタ)。

---

## tour への取り込み案

§10 として「**面白かった feature 集**」のような新設章にする、もしくは §3 (16503 の隣) と §5 (Concept search) の間に「§4.5 feature の典型バリエーション」を挟むなど。**全部入れると重い** ので、tour 本編には:

- **A から 1〜2 個**(feat #2371 か #28927 — Python の超 narrow さの代表)
- **B から 1 個**(feat #6460 — 「同じ単語でも文脈で別 feature」のキレが良い)
- **C から 1 個**(feat #7690 か feat #17402 — 日本語の機能語が独立 feature になる驚き)
- **D から 1 個**(feat #25342 — cross-lingual 概念の実例。tour 既出の 16504 と並べると説得力増)
- **E から 1 個**(feat #5657 か feat #18310 — トークナイザの粒度との関係)

の **計 5〜6 個** くらいが良いバランス。残りはこのファイル `feature_findings.md` に残しておけば、note 記事化時に深掘りで使える。
