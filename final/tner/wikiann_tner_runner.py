from __future__ import annotations

import re
import warnings
from typing import Dict, Iterable, List, Sequence, Tuple

from tner import TransformersNER

_SENT_RE = re.compile(r"[^.!?]+[.!?]?")
_WS = set(" \t\n\r\f\v")

# Common sub‑word prefixes across tokenisers
_PREFIXES = ("Ġ", "▁", "##")

# ANSI Formatting escape sequences
ANSI_BOLD = "\033[1m"
ANSI_BLUE = "\033[94m"
ANSI_RESET = "\033[0m"


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Return **absolute** ``(start, end)`` indices for every sentence.

    Leading whitespace after a sentence boundary is **stripped**, so each span
    starts exactly at the first non‑space character of its sentence.
    """
    spans: List[Tuple[int, int]] = []
    for m in _SENT_RE.finditer(text):
        s, e = m.start(), m.end()
        # Trim any leading spaces/tabs/newlines that sneak in after the period
        while s < e and text[s] in _WS:
            s += 1
        spans.append((s, e))
    return spans


def _clean(tok: str) -> str:
    while tok.startswith(_PREFIXES):
        tok = tok[1:]
    return tok


def _spans_from_offset(sentence: str, *, sent_offset: int, tokenizer) -> List[Tuple[int, int]] | None:
    """Return spans via `offset_mapping` if tokenizer supports it; else *None*."""
    if callable(tokenizer):
        try:
            enc = tokenizer(
                sentence,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
            )
            if "offset_mapping" in enc:
                return [
                    (s + sent_offset, e + sent_offset) for s, e in enc["offset_mapping"]
                ]
        except Exception:
            pass
    return None


def _spans_by_search(sentence: str, *, tokens: Sequence[str], sent_offset: int) -> List[Tuple[int, int]]:
    """Greedy deterministic search mapping tokens → char spans."""
    spans: List[Tuple[int, int]] = []
    i, n = 0, len(sentence)
    for raw in tokens:
        tok = _clean(raw)
        if not tok:
            spans.append((i + sent_offset, i + sent_offset))
            continue
        pos = sentence.find(tok, i)
        if pos == -1:
            pos = sentence.lower().find(tok.lower(), i)
        if pos == -1:
            raise ValueError(f"Token '{raw}' not found after index {i}:\n{sentence!r}")
        start = pos
        end = pos + len(tok)
        spans.append((start + sent_offset, end + sent_offset))
        i = end
        if i > n:
            raise AssertionError("Token alignment overflow – check input text.")
    return spans


def _render(text: str, ent_spans: Iterable[Tuple[int, int, str]]) -> str:
    out, last = [], 0
    for s, e, lbl in ent_spans:
        out.append(text[last:s])
        out.append(f"{ANSI_BOLD}{text[s:e]}{ANSI_RESET}[{ANSI_BLUE}{lbl}{ANSI_RESET}]")
        last = e
    out.append(text[last:])
    return "".join(out)


def visualize_entities(text: str, model: TransformersNER, min_prob: float | None = None) -> None:
    """Print *text* with entities highlighted (ANSI).

    Parameters
    ----------
    text : str
        Raw input text.
    model : TransformersNER
        A TNER model instance.
    min_prob : float, optional
        Drop entities whose mean token probability is below this value.
    """
    tokenizer = getattr(model, "tokenizer", None)

    # 1. Sentence segmentation (whitespace‑trimmed)
    sent_spans = _sentence_spans(text)
    sentences = [text[s:e] for s, e in sent_spans]

    # 2. Model inference
    output: Dict = model.predict(sentences)

    # 3. Build absolute entity spans
    ent_spans: List[Tuple[int, int, str, float]] = []  # (start, end, label, prob)
    for i_sent, (sent_start, _) in enumerate(sent_spans):
        spans = _spans_from_offset(sentences[i_sent], sent_offset=sent_start, tokenizer=tokenizer)
        if spans is None:
            spans = _spans_by_search(
                sentences[i_sent], tokens=output["input"][i_sent], sent_offset=sent_start
            )
        for ent in output["entity_prediction"][i_sent]:
            pos = ent["position"]
            start_char = spans[pos[0]][0]
            end_char = spans[pos[-1]][1]
            prob = float(sum(ent["probability"]) / len(ent["probability"]))
            ent_spans.append((start_char, end_char, ent["type"], prob))

    # 4. Probability filter
    if min_prob is not None:
        before = len(ent_spans)
        ent_spans = [t for t in ent_spans if t[3] >= min_prob]
        if (removed := before - len(ent_spans)):
            warnings.warn(
                f"{removed} entities removed (prob < {min_prob:.2f}).",
                RuntimeWarning,
                stacklevel=2,
            )

    # 5. Render – ignore prob in output
    ent_spans.sort(key=lambda t: t[0])
    print(_render(text, ((s, e, l) for s, e, l, _ in ent_spans)))


def get_lines_set(file_path: str):
    """Return a *set* of non‑empty, stripped lines from *file_path*."""
    with open(file_path, "r", encoding="utf-8") as fh:
        return {ln.strip() for ln in fh if ln.strip()}


## English

en_text = """
World War II or the Second World War (1 September 1939 – 2 September 1945) was a global conflict between two coalitions: the Allies and the Axis powers. Nearly all the world's countries—including all the great powers—participated, with many investing all available economic, industrial, and scientific capabilities in pursuit of total war, blurring the distinction between military and civilian resources. Tanks and aircraft played major roles, with the latter enabling the strategic bombing of population centres and delivery of the only two nuclear weapons ever used in war. World War II was the deadliest conflict in history, resulting in 70 to 85 million deaths, more than half being civilians. Millions died in genocides, including the Holocaust of European Jews, as well as from massacres, starvation, and disease. Following the Allied powers' victory, Germany, Austria, Japan, and Korea were occupied, and war crimes tribunals were conducted against German and Japanese leaders.
The causes of World War II included unresolved tensions in the aftermath of World War I and the rise of fascism in Europe and militarism in Japan. Key events leading up to the war included Japan's invasion of Manchuria, the Spanish Civil War, the outbreak of the Second Sino-Japanese War, and Germany's annexations of Austria and the Sudetenland. World War II is generally considered to have begun on 1 September 1939, when Nazi Germany, under Adolf Hitler, invaded Poland, prompting the United Kingdom and France to declare war on Germany. Poland was divided between Germany and the Soviet Union under the Molotov–Ribbentrop Pact, in which they had agreed on "spheres of influence" in Eastern Europe. In 1940, the Soviets annexed the Baltic states and parts of Finland and Romania. After the fall of France in June 1940, the war continued mainly between Germany and the British Empire, with fighting in the Balkans, Mediterranean, and Middle East, the aerial Battle of Britain and the Blitz, and naval Battle of the Atlantic. Through a series of campaigns and treaties, Germany took control of much of continental Europe and formed the Axis alliance with Italy, Japan, and other countries. In June 1941, Germany led the European Axis in an invasion of the Soviet Union, opening the Eastern Front and initially making large territorial gains.
Japan aimed to dominate East Asia and the Asia-Pacific, and by 1937 was at war with the Republic of China. In December 1941, Japan attacked American and British territories in Southeast Asia and the Central Pacific, including Pearl Harbor in Hawaii, which resulted in the US and the UK declaring war against Japan, and the European Axis declaring war on the US. Japan conquered much of coastal China and Southeast Asia, but its advances in the Pacific were halted in mid-1942 after its defeat in the naval Battle of Midway; Germany and Italy were defeated in North Africa and at Stalingrad in the Soviet Union. Key setbacks in 1943—including German defeats on the Eastern Front, the Allied invasions of Sicily and the Italian mainland, and Allied offensives in the Pacific—cost the Axis powers their initiative and forced them into strategic retreat on all fronts. In 1944, the Western Allies invaded German-occupied France at Normandy, while the Soviet Union regained its territorial losses and pushed Germany and its allies westward. At the same time, Japan suffered reversals in mainland Asia, while the Allies crippled the Japanese Navy and captured key islands.
The war in Europe concluded with the liberation of German-occupied territories; the invasion of Germany by the Western Allies and the Soviet Union, culminating in the fall of Berlin to Soviet troops; Hitler's suicide; and the German unconditional surrender on 8 May 1945. Following the refusal of Japan to surrender on the terms of the Potsdam Declaration, the US dropped the first atomic bombs on Hiroshima and Nagasaki on 6 and 9 August. Faced with an imminent invasion of the Japanese archipelago, the possibility of further atomic bombings, and the Soviet declaration of war against Japan and its invasion of Manchuria, Japan announced its unconditional surrender on 15 August and signed a surrender document on 2 September 1945, marking the end of the war.
World War II changed the political alignment and social structure of the world, and it set the foundation of international relations for the rest of the 20th century and into the 21st century. The United Nations was established to foster international cooperation and prevent conflicts, with the victorious great powers—China, France, the Soviet Union, the UK, and the US—becoming the permanent members of its security council. The Soviet Union and the United States emerged as rival superpowers, setting the stage for the Cold War. In the wake of European devastation, the influence of its great powers waned, triggering the decolonisation of Africa and Asia. Most countries whose industries had been damaged moved towards economic recovery and expansion.
"""

# en_text = "The war in Europe concluded with the liberation of German-occupied territories; the invasion of Germany by the Western Allies and the Soviet Union, culminating in the fall of Berlin to Soviet troops; Hitler's suicide; and the German unconditional surrender on 8 May 1945."
en_text = "The war in Europe concluded with the liberation of German-occupied territories; the invasion of Germany by the Western Allies and the Soviet Union, culminating in the fall of Berlin to Soviet troops; Hitler's suicide; and the German unconditional surrender on 8 May 1945. Following the refusal of Japan to surrender on the terms of the Potsdam Declaration, the US dropped the first atomic bombs on Hiroshima and Nagasaki on 6 and 9 August. Faced with an imminent invasion of the Japanese archipelago, the possibility of further atomic bombings, and the Soviet declaration of war against Japan and its invasion of Manchuria, Japan announced its unconditional surrender on 15 August and signed a surrender document on 2 September 1945, marking the end of the war."
# en_text = "Following the refusal of Japan to surrender on the terms of the Potsdam Declaration, the US dropped the first atomic bombs on Hiroshima and Nagasaki on 6 and 9 August. Faced with an imminent invasion of the Japanese archipelago, the possibility of further atomic bombings, and the Soviet declaration of war against Japan and its invasion of Manchuria, Japan announced its unconditional surrender on 15 August and signed a surrender document on 2 September 1945, marking the end of the war."

# visualize_entities(en_text, model)

## Bulgarian

# bg_text = """
# Втората световна война е глобална война, която продължава от 1 септември 1939 г. до 2 септември 1945 г. и се превръща в най-мащабния военен конфликт в историята на човечеството.
# Почти всички страни в света, включително всички Велики сили, в определен момент се включват във войната, присъединявайки се към един от двата противостоящи военни съюза – Съюзниците и Тристранния пакт. Стига се до състояние на тотална война, в която пряко участват над 100 милиона души от 62 страни от общо 74 по това време. Основните участници влагат целия си стопански, промишлен и научен потенциал във военното усилие, размивайки границите между граждански и военни ресурси. Втората световна война е най-смъртоносният и кръвопролитен конфликт в човешката история, довел до между 70 и 85 милиона жертви, повечето от които цивилни в Съветския съюз и Китай. Войната е съпътствана от кланета, геноцида на Холокост, стратегически бомбардировки, умишлено предизвикани масов глад и епидемии, както и единственото в историята използване на ядрено оръжие при военни действия.
# Япония, в стремежа си към хегемония в Далечния Изток, се намира в състояние на война с Китай още от 1937 година, но обикновено за начална дата на Втората световна война се приема 1 септември 1939 година, датата на нападението на Германия срещу Полша, последвано от обявяването на война на Германия от Франция и Великобритания. От края на 1939 до началото на 1941 година, чрез поредица от военни кампании и дипломатически споразумения, Германия завладява или установява контрол над по-голямата част от континентална Европа. С Пакта „Рибентроп-Молотов“ Германия и Съветският съюз си поделят и анексират територии от своите съседни държави Полша, Финландия, Румъния и Прибалтийските страни. След Северноафриканската и Източноафриканската кампания и падането на Франция в средата на 1940 година, войната в Европа се води главно между Тристранния пакт (Оста) и Великобритания. Следват Балканската кампания, въздушната битка за Британия и продължителната морска битка за Атлантика.
# На 22 юни 1941 година европейските сили на Оста нападат Съветския съюз, откривайки най-обширния сухопътен военен театър в историята. Образувалият се Източен фронт блокира Тристранния пакт, най-вече германския Вермахт, в тежка изтощителна война. През декември 1941 година Япония напада изненадващо Съединените американски щати и западните владения в Тихоокеанския регион, което довежда до включването на Съединените щати във войната срещу Тристранния пакт. Последвалото бързо японско настъпление в западната част на Тихия океан е прието от много азиатци като освобождение от западната хегемония.
# Напредъкът на Тристранния пакт на Тихоокеанския театър е спрян през 1942 година, когато Япония губи ключовата битка при Мидуей. Малко по-късно Германия и Италия претърпяват поражение в Северна Африка, а след това и решителен разгром при Сталинград в Съветския съюз. Ключови неуспехи през 1943 година, сред които поредица поражения на Източния фронт, десанти на Съюзниците в Сицилия и Италия и победи на Съюзниците на Тихоокеанския театър отнемат инициативата от Тристранния пакт и предизвикват стратегическото му отстъпление по всички фронтове. През 1944 година Западните Съюзници навлизат във Франция, докато Съветският съюз възвръща загубените си територии и настъпва срещу Германия и нейните съюзници. През 1944 и 1945 година японците претърпяват тежки поражения в континентална Азия – в Централен и Южен Китай и в Бирма, докато Съюзниците нанасят сериозни поражения и загуби на японския флот и завземат стратегически важни острови.
# Войната в Европа завършва с настъпление на Съюзниците в Германия, достигнало своята кулминация с превземането на Берлин от съветските войски, самоубийството на Адолф Хитлер (според хора е лайно) безусловната капитулация на Германия на 8 май 1945 година. След американските атомни бомбардировки в Хирошима и Нагасаки на 6 и 9 август и изправена пред заплахата от десант на Японските острови, нови атомни бомбардировки и война със Съветския съюз, на 15 август 1945 година Япония обявява намерението си да капитулира, с което Съюзниците постигат пълна победа и в Азия.
# Втората световна война променя политическите отношения и обществените структури по целия свят. Създадена е Организацията на обединените нации, чиято цел е да насърчава международното сътрудничество и да предотвратява бъдещи конфликти, а Великите сили победителки – Великобритания, Китай, Съветският съюз, Съединените щати и Франция – стават постоянни членки на нейния Съвет за сигурност. Съветският съюз и Съединените щати се превръщат в свръхдържави съперници, подготвяйки условията за продължилата половин век Студена война. След разрушенията в Европа влиянието на нейните Велики сили отслабва, довеждайки до деколонизацията на Африка и Азия. Въпреки това повечето страни бързо преодоляват стопанския срив от войната и преживяват икономическо възстановяване и подем. Политическата интеграция, най-вече в Европа, започва като усилие за предотвратяване на бъдещи военни действия, прекратяване на довоенните съперничества и изграждане на чувство за обща идентичност. """
#
# visualize_entities(bg_text, model)

## Czech

# cs_text = """
# Druhá světová válka byl globální vojenský konflikt v letech 1939–1945, jehož se zúčastnila většina států světa. Tento konflikt se stal s více než 62 miliony obětí na životech dosud největším a nejvíce zničujícím válečným střetnutím v dějinách lidstva.
# Brzy po okupaci zbytku Československa 15. března 1939 vypukla válka v Evropě. Začala dne 1. září 1939, když nacistické Německo přepadlo Polsko. Krátce poté vyhlásily Francie, Spojené království a státy Commonwealthu Německu válku. 17. září napadl Polsko i Sovětský svaz (SSSR). Německé invazi do Polska předcházela jeho smlouva o neútočení se SSSR, takzvaný pakt Ribbentrop–Molotov, podepsaná 23. srpna 1939. V tajném protokolu k tomuto paktu si tyto dva státy dočasně rozdělily sféry vlivu tzv. demarkační Curzonovu linii. Byla vytyčena v roce 1919 mezi druhou polskou republikou a Sovětským svazem, dvěma novými státy, které vznikly po první světové válce. Curzonova linie vznikla jako diplomatický podklad pro budoucí dohodu o hranicích. Vytyčila ji Rada Dohody podle usnesení z 8. prosince 1919. V důsledku toho byl de facto uvolněn prostor pro vojenskou expanzi obou stran. Ze strany Německa byl na části území Polska až k demarkační linii vytvořen Generalgouvernement – jako správní jednotka utvořená 26.10.1939 na základě Hitlerova dekretu, která zahrnovala část okupovaného území původního meziválečného Polska, které nebylo začleněno do Třetí říše (vojvodství Kielecké, Krakovské, Lublinské, část Lodžského a Varšavského). Sídlem správy byl Krakov. V čele Generálního gouvernementu stál válečný zločinec Hans Frank, pod jehož vedením docházelo k brutálnímu útisku a cílenému vyhlazování nejen polského, ale i židovského obyvatelstva v Polsku.
# Blesková válka na západě Evropy začala dne 10. května 1940, kdy německý Wehrmacht na rozkaz Adolfa Hitlera překročil hranice Belgie, Nizozemska a Lucemburska, a poté obešel obrannou Maginotovu linii. Po rychlé porážce francouzské armády vstoupila německá vojska 14. června do Paříže. Francie kapitulovala 22. června 1940 a do listopadu 1942 německá armáda postupně obsadila celou zemi.
# Sovětský svaz se stal jedním ze Spojenců druhé světové války proti Ose Berlín–Řím–Tokio poté, co nacistické Německo zahájilo dne 22. června 1941 proti němu rozsáhlou a ničivou operaci Barbarossa. Zprvu musela Rudá armáda ustupovat až k Moskvě. Po urputných bojích, např. v bitvě u Stalingradu (podzim 1942 – zima 1943), o Kurský oblouk, či obležení Leningradu (dnešní Petrohrad) atd., začala sovětská vojska zatlačovat Wehrmacht západním směrem a dne 2. května 1945 dobyla Berlín.
# Dne 7. července 1937 se udál incident na mostě Marca Pola v Pekingu. Tímto relativně malým vojenským střetnutím de facto vznikla druhá čínsko-japonská válka, zprvu bez formálního válečného stavu mezi Japonským císařstvím a Čínskou republikou. Japonsko poté pokračovalo ve své rozsáhlé expanzi proti čínským územím, pak přepadlo a dobylo řadu zemí v jihovýchodní Asii. Dne 7. prosince 1941 zaútočila letadla z japonských letadlových lodí na americkou námořní základnu Pearl Harbor na Havajských ostrovech. Den poté vstoupily Spojené státy americké do války proti Japonsku. Teprve 9. prosince 1941 vyhlásila čínská vláda oficiálně Japonsku válku. 11. prosince 1941 vyhlásily nacistické Německo a Itálie válku Spojeným státům, čímž byl utvrzen stav globálního konfliktu.
# Konec války v Evropě nastal 8. května 1945 bezpodmínečnou kapitulací Německa. Po americkém svržení atomových bomb ve dnech 6. a 9. srpna 1945 na města Hirošima a Nagasaki kapitulovalo Japonsko 2. září 1945.
# Příčiny války bývají hledány v důsledcích ideologií a politických směrů, jako jsou nacionalismus a imperialismus. Podle některých historiků byla jednou z hlavních příčin nespokojenost vládnoucích kruhů Německa s dopady Versailleské smlouvy, která měla prohloubit pocit ponížení po prohrané první světové válce, a v následcích velké hospodářské krize na přelomu dvacátých a třicátých let. Tyto vlivy zásadním způsobem oslabily mnoho evropských států, čímž umožnily vzestup nacismu a italského fašismu.
# Druhou světovou válku provázely v dosud nevídané míře válečné zločiny, zločiny proti lidskosti a nehumánní zacházení s válečnými zajatci, zvláště se sovětskými vojáky ze strany Německa. Průběhem bojů bylo podstatně zasaženo rovněž civilní obyvatelstvo, jež utrpělo obrovské ztráty. Nejhorším příkladem genocidy se stal holokaust (šoa), kterému na základě nacistické rasové ideologie padlo za oběť šest milionů Židů v koncentračních táborech a na jiných místech v mnoha zemích Evropy. Masakr čínského obyvatelstva, který spáchali Japonci v Nankingu v prosinci 1937, byl jedním z největších zločinů. V rámci operace Intelligenzaktion v roce 1939 zavraždili němečtí nacisté 60 000 až 100 000 příslušníků polské inteligence, důstojníků a představitelů měst i státu. V roce 1940 provedla sovětská NKVD tzv. katyňský masakr, při kterém bylo povražděno přibližně 22 000 polských důstojníků a příslušníků inteligence. Milionové oběti utrpělo slovanské civilní obyvatelstvo – Rusové, Ukrajinci, Bělorusové, Poláci a jiní – na územích východní fronty, kde ztratilo životy osm milionů lidí. Ti podléhali nemocem a hladu vyvolaným válečnými operacemi a masakrům páchaným na územích obsazených Wehrmachtem a jednotkami Waffen-SS.
# Válečné úsilí pohlcovalo téměř veškerý lidský, ekonomický, průmyslový a vědecký potenciál všech zúčastněných národů. Mnoho států utrpělo nepředstavitelné materiální ztráty a devastaci svého kulturního dědictví. Lze proto hovořit o tom, že se jednalo o totální válku. Téměř všechny zúčastněné strany se v menší či větší míře odchylovaly od požadavku vést válku „civilizovanými metodami“. I když Spojené království v roce 1940 odmítalo plošné nálety na nepřátelská města, posléze se k nim spolu se Spojenými státy samo uchýlilo.
# V samotném závěru světové války byla ustavena Organizace spojených národů, jejímž ústředním cílem byla a je i v současnosti prevence vzniku dalších válečných konfliktů. Po skončení války upevnily vítězné mocnosti USA a SSSR své postavení dvou světových supervelmocí. Jejich stále větší vzájemný antagonismus vedl k bipolárnímu rozdělení světa a k počátku studené války. První generální tajemník Komunistické strany Sovětského svazu Josif Stalin spustil napříč evropským kontinentem tzv. železnou oponu, která od sebe oddělila západní svět a státy ve Východní Evropě, především z obavy vojenského obsazení zemí Východní Evropy. Vedlejším efektem války byl také vzrůst požadavků na právo na sebeurčení mezi národy ovládanými koloniálními mocnostmi, což vedlo k akceleraci dekolonizačních hnutí v Asii a v Africe.
# Jednalo se o nejrozsáhlejší válku v dějinách, které se přímo účastnilo více než 100 milionů lidí z více než 30 zemí. Ve stavu totální války vrhli hlavní účastníci do válečného úsilí veškeré své hospodářské, průmyslové a vědecké kapacity, čímž se smazaly rozdíly mezi civilním a vojenským obyvatelstvem.
# """
#
# visualize_entities(cs_text, model)

# ru_text = """
# Вторая мировая война (1 сентября 1939 — 2 сентября 1945) — война двух мировых военно-политических коалиций, ставшая крупнейшим вооружённым конфликтом в истории человечества. В этой войне участвовали 62 государства из 74 существовавших на тот момент (80 % населения Земного шара). Боевые действия велись на территории Евразии и Африки и в водах всех океанов. Это единственный конфликт, в котором было применено ядерное оружие. В результате войны погибло более 70 миллионов человек, большинство из которых — мирные жители.
# Началом Второй мировой войны считается 1 сентября 1939 года, когда нацистская Германия начала вторжение в Польшу. За этим 3 сентября последовало объявление войны Германии со стороны Великобритании и Франции. 17 сентября с востока в Польшу вторглись войска СССР, а в октябре страна по подписанному в августе 1939 года пакту Молотова — Риббентропа была поделена между Германией и СССР. За этим последовало стремительное завоевание Германией Дании и Норвегии (апрель — июнь 1940), Франции и стран Бенилюкса (май — июнь 1940), Югославии (апрель 1941). С июня 1940 года в войне участвовала фашистская Италия, помогавшая Германии во Франции и объявившая обернувшуюся неудачей войну Греции 28 октября 1940 года. В апреле 1941 года Италия и Германия совместно смогли разбить греческую оборону и оккупировать страну. Некоторые страны добровольно стали союзниками Германии и Италии: такими являлись Венгрия, Румыния и Болгария, а также в некоторой степени Финляндия. К июню 1941 года единственным серьёзным противником Германии и Италии оставалась лишь Великобритания, попытка ослабить её не увенчалась успехом.
# 22 июня 1941 года Германия и её союзники осуществили вторжение в СССР (который к этому моменту успел захватить и присоединить к себе части Карелии, страны Прибалтики, Бессарабию и Северную Буковину). Великая Отечественная война стала решающей в дальнейшем развитии событий Второй мировой войны в Европе, а Советский Союз понёс наибольшие потери в этой войне.Тем временем, боевые действия шли и в Азии. Главным очагом агрессии там стала Японская империя, желавшая доминировать в этом регионе. Ещё с 1931 года она несколько раз атаковала соседний Китай, в котором шла гражданская война. 7 июля 1937 года Япония развязала против Китая полномасштабную войну. 7 декабря 1941 года японская авиация и флот атаковали Перл-Харбор, что привело к вступлению в войну против стран «Оси» Соединённых Штатов Америки — возник Тихоокеанский театр военных действий Второй мировой войны. На стороне Японии в боевых действиях участвовал и Таиланд. Продвижение японской армии было остановлено после поражения в битве за Мидуэй в июне 1942 года, после чего Япония один за другим стала терять тихоокеанские острова. Одновременно с этим начались поражения и в Европе: в 1942—1944 годах Советский Союз одержал ряд военных побед, ослабивших германскую армию: Сталинградская битва 1942—1943 годов, Курская битва 1943 года и несколько военных операций 1944 года привели к освобождению почти всей оккупированной территории страны и переходу на сторону антигитлеровской коалиции Финляндии, Румынии и Болгарии. Одновременно с этим Италия и Германия были вытеснены из Северной Африки. В Италии произошёл государственный переворот, Муссолини был арестован, новое правительство Италии заявило о выходе из войны и запросило перемирие. В июне 1944 года войска США, Великобритании, и военные формирования Свободной Франции, открыли второй фронт в Европе, высадившись в Нормандии и затем освободив Францию и другие страны Западной Европы.В 1945 году произошёл окончательный коллапс стран «оси». Вторжение англо-американских войск в Германию и Битва за Берлин вывели из войны нацистскую Германию, подписавшую акт о капитуляции 8 мая (по московскому времени 9 мая) 1945 года. В июле 1945 года США отправили Японии Потсдамскую декларацию, которую та отклонила. Вслед за этим последовали атомные бомбардировки Хиросимы и Нагасаки и вступление в войну СССР. 2 сентября 1945 года Япония подписала акт о капитуляции. На этом Вторая мировая война завершилась.По результатам войны, кроме территориальных изменений и репараций, впервые в мировой истории был создан международный трибунал для осуждения бывшего руководства стран признанных агрессорами и виновниками развязывания войны, в рамках вновь созданных статей международного права, ставших обязательными к исполнению и впредь. Главные виновные, признанные в развязывании войны — были повешены, остальные были осуждены на различные сроки заключения. Была создана новая организация, призванная обеспечить длительный мир в Европе (а по возможности, и во всём мире) — Организация Объединённых Наций (ООН), деятельность не оправдавшей себя Лиги Наций была прекращена.
# """
# visualize_entities(ru_text, model)

# sk_text = """
# Druhá svetová vojna je dodnes najväčší a najrozsiahlejší ozbrojený konflikt v dejinách ľudstva, ktorý stál život asi 45 až 60 miliónov ľudí. Boje prebiehali v Európe, Ázii, Afrike a Tichomorí a zúčastňovali sa na nich muži i ženy aj z oboch ďalších obývaných kontinentov: Ameriky a Austrálie. Počas šiestich rokov trvania zahynuli alebo zomreli v dôsledku vojnových útrap desiatky miliónov civilistov, milióny príslušníkov ozbrojených síl, boli zničené celé mestá a spôsobené nevyčísliteľné škody na majetku a kultúrnom dedičstve ľudstva. Príčiny vojny v Európe možno hľadať v napätí vyvolanom chybne koncipovanou Versaillskou zmluvou, negatívnych dopadoch Veľkej hospodárskej krízy na prelome 20. a 30. rokov, ktorá kriticky oslabila všetky štáty a ich vlády ako aj slabosť Spoločnosti národov a mocností, ktoré mali udržovať svetový mier a dohliadať na dodržovanie versaillského systému, čo v Nemecku umožnilo vzostup nacistického režimu pod vedením Adolfa Hitlera a jeho stúpencov.
# Na ázijskom bojisku sa za začiatok vojny považuje japonská invázia do Číny (7. júl 1937), ale niektoré zdroje uvádzajú oveľa skorší dátum: rok 1931, keď Japonsko vtrhlo do Mandžuska. V Európe sa boje druhej svetovej vojny začali 1. septembra 1939, keď nemecký Wehrmacht vpadol do Poľska[2], krátko na to 17. septembra 1939 napadol Poľsko aj Sovietsky zväz. Západné demokracie Spojené kráľovstvo a Francúzsko vyhlásili Nemecku vojnu, ale obmedzili sa na menšie akcie, ktoré Nemecko nezastavili (tzv. čudná vojna). 9. apríla 1940 Nemecko napadlo Dánsko a Nórsko a postupne ich obsadilo. 10. mája 1940 Nemecko napadlo cez krajiny Beneluxu aj Francúzsko, ktorého obrana sa zakrátko zrútila. 22. júna 1940 Francúzsko kapitulovalo. 22. júna 1941 Nemecko napadlo Sovietsky zväz (operácia Barbarossa), ktorý sa pridal na stranu Spojencov. Spojené štáty, ktoré už predtým pomáhali Spojeneckým krajinám, sa do vojny zapojili 7. decembra 1941 po japonskom útoku na námornú základňu USA v Pearl Harbor. Po ťažkých bitkách pri Stalingrade a Kursku na východnom fronte boli nemecké vojská prinútené ustupovať. Západní spojenci znovu otvorili tzv. druhý front 6. júna 1944. Po porážkach Nemecka vo Francúzsku, Bielorusku a Poľsku Spojenci vstúpili na územie Nemecka. Koniec vojny v Európe nastal 8. mája 1945 kapituláciou Nemecka. V Ázii kapitulovalo Japonsko 2. septembra toho istého roku po americkom zhodení dvoch atómových bômb (Little Boy a Fat Man) na mestá Hirošima a Nagasaki.
# Medzi dôsledky druhej svetovej vojny patrí vytvorenie dvoch blokov: západného bloku, ktorý sa sformoval do organizácie NATO, a východného bloku, ktorý na seba vzal podobu Varšavskej zmluvy tvorenej najmä východoeurópskymi socialistickými krajinami pod vedením Sovietskeho zväzu. Vzťahy medzi týmito dvoma blokmi boli značne napäté a čoskoro prerástli do tzv. Studenej vojny, ktorá sa okrem politických bojov prejavila aj v niektorých vojenských konfliktoch (Kórejská vojna, Vietnamská vojna, Arabsko-izraelské vojny atď.)
# """
#
# visualize_entities(sk_text, model)
#
# sl_text = """
# Druga svetovna vojna je bila najobsežnejši in najdražji oborožen spopad v zgodovini. Potekal je v letih od 1939 do 1945, v njem pa je sodelovala večina svetovnih držav z več kot 100 milijonov pripadnikov oboroženih sil. Boj je potekal večinoma med Združenim Kraljestvom, Francijo, Sovjetsko zvezo, Kitajsko in Združenimi državami Amerike proti Nemčiji, Italiji in Japonski, oziroma med zavezniki in silami osi. Odvijal se je hkrati po celem svetu, zahteval pa je približno 60 milijonov človeških življenj. Zaradi slednjega ter zaradi visokega deleža mrtvih civilistov, pri čemer izstopata holokavst ter jedrsko bombardiranje Hirošime in Nagasakija, se drugo svetovno vojno označuje kot najbolj krvav spopad v človeški zgodovini.[1][2]. Tem žrtvam je treba dodati še približno 6 milijonov pobitih v povojnih pobojih, ki so prav tako posledica fašizma kot vzroka 2. svetovne vojne. V povprečju je bilo po vojni usmrčenih 10 % na število žrtev med vojno.
# V splošnem je sprejeto, da se je vojna začela 1. septembra 1939 z nemško invazijo na Poljsko in posledično z napovedjo vojne Nemčiji s strani Združenega Kraljestva, Francije in večino držav Britanskega imperija ter Commonwealtha. Kitajska in Japonski imperij sta bila ob začetku že v vojni,[3] medtem ko so se druge države, ki sprva niso bile vključene v vojno, pridružile kasneje zaradi določenih dogodkov, kot sta nemška invazija na Sovjetsko zvezo ter japonski napad na ameriško pomorsko oporišče Pearl Harbor, kar je sprožilo vojno napoved Japonski s strani ZDA, Commonwealtha[4] ter Nizozemske.[5]
# Vojna se je končala z zmago zaveznikov nad Nemčijo in Japonsko leta 1945. Kot posledica vojne sta se politično prepričanje ter družbena struktura močno spremenila. Medtem ko je bila ustanovljena Organizacija združenih narodov (OZN) za krepitev mednarodnega sodelovanja ter preprečevanje nadaljnjih spopadov, se je zaradi ideoloških razlik med takratnima supersilama, tj. med ZDA in Sovjetsko zvezo, začelo obdobje hladne vojne. V tem času je OZN-ovo zagovarjanje pravice narodov do samoodločbe pospešilo dekolonizacijska gibanja v Aziji in Afriki, v Zahodni Evropi pa sta se gospodarstvo ter proces Evropska integracije okrepila.
# """
#
# visualize_entities(sl_text, model)

uk_text = """
Друга світова війна — глобальний збройний конфлікт, що тривав від 1 вересня 1939 року до 2 вересня 1945 року. У війні взяло участь понад 60 країн, зокрема всі великі держави, які утворили два протилежні військові табори: блок країн Осі та антигітлерівську коаліцію («союзники»). Безпосередню участь у бойових діях брали понад 100 мільйонів осіб. Супротивні держави кинули всі економічні, промислові та наукові можливості на потреби фронту, стираючи різницю між цивільними та військовими ресурсами. Загальні людські втрати коливаються між 50 й 80 мільйонами осіб, більшість із яких були мешканцями Радянського Союзу та Китаю. Друга світова війна відзначилася численними масовими вбивствами і злочинами проти людяності, насамперед Голокостом, стратегічними килимовими бомбардуваннями та єдиним в історії військовим застосуванням ядерної зброї.
Основними причинами війни стали політичні суперечності, породжені недосконалою Версальською системою, та агресивна експансіоністська політика нацистської Німеччини, Японської імперії та Італії. 1 вересня 1939 року гітлерівські війська вторглися в Польщу. 3 вересня Велика Британія та Франція оголосили Німеччині війну. Упродовж 1939—1941 років завдяки серії успішних військових кампаній та низки дипломатичних заходів Німеччина захопила більшу частину континентальної Європи. Саме тоді й Радянський Союз анексував (повністю або частково) території сусідніх європейських держав: Польщі, Румунії, Фінляндії та країн Балтії, що відійшли до його сфери впливу на підставі Пакту Молотова — Ріббентропа. Після початку бойових дій у Північній Африці та падіння Франції в середині 1940 року війна тривала насамперед між країнами Осі та Великою Британією, повітряні сили якої зуміли відбити німецькі повітряні атаки. У цей же час бойові дії поширились на Балканський півострів та Атлантичний океан. Японія окупувала частину Китаю та Південно-Східної Азії, взявши під контроль важливі джерела сировини.
22 червня 1941 року війська країн Осі чисельністю 3.5 мільйонів осіб вторглися в Радянський Союз, маючи на меті завоювання «життєвого простору» в Східній Європі. Відкривши найбільший в історії сухопутний фронт, німецькі війська спершу доволі швидко окупували західні регіони СРСР, однак в битві за Москву зазнали поразки. В цей же час Японія віроломно напала на США та підкорила західну частину Тихого океану. Задля протистояння агресії країн Осі створено Антигітлерівську коаліцію 26 країн, в окупованих країнах розгорнувся рух опору. У лютому 1943 радянська армія здобула перемогу під Сталінградом. У Північній Африці німецькі та італійські війська зазнали поразки під Ель-Аламейном. Просування Японії зупинили сили американців і австралійців у битві за Мідвей. У 1943 році після низки військових невдач Гітлера на Східному фронті, висадки союзників у Сицилії та Італії, що призвело до капітуляції останньої, і перемог США на Тихому океані, країни Осі втратили ініціативу та перейшли до стратегічного відступу на всіх фронтах. У 1944 році армії західних альянтів визволили Західну та Центральну Європу, у той час як радянські війська вигнали війська Німеччини та окупантів з власної території та країн Східної й Південно-Східної Європи.
Протягом 1944 та 1945 років Японія зазнала великих втрат у материковій Азії, у Південному Китаї та Бірмі; союзники знищили японський флот і заволоділи ключовими островами в західній частині Тихого океану. Німеччина опинилася в щільному кільці. До кінця квітня 1945 року радянські війська заволоділи значною частиною її території, зокрема й Берліном; Адольф Гітлер вчинив самогубство. 8 травня керівництво Вермахту підписало Акт про беззастережну капітуляцію. Ця дата вважається Днем перемоги над нацизмом в Європі. Після опублікування 26 липня 1945 Потсдамської декларації та відмови Японії капітулювати на її умовах США скинули атомні бомби на міста Хіросіму і Нагасакі 6 і 9 серпня відповідно. У серпні 1945 Радянський Союз розгорнув бойові дії проти Японії. Неминуче вторгнення американців на японський архіпелаг, а також можливість інших атомних бомбардувань змусили керівництво цієї острівної країни здатися. Акт про капітуляцію Японії підписали 2 вересня 1945 року на борту лінкора «Міссурі». Війна в Азії закінчилась, закріпивши загальну перемогу Антигітлерівської коаліції.
Друга світова стала наймасштабнішою та найкривавішою війною в історії людства, великим переламом XX століття, що докорінно змінив політичну карту і соціальну структуру світу. Для сприяння розвитку міжнародного співробітництва та запобігання майбутнім конфліктам створено Організацію Об'єднаних Націй. Післявоєнний порядок утвердив гегемонію Сполучених Штатів і Радянського Союзу, суперництво яких призвело до утворення капіталістичного й соціалістичного таборів та початку Холодної війни. Світовий вплив європейських держав значно ослаб, почався процес деколонізації Азії та Африки. Перед країнами, чиї галузі економіки були знищені, гостро стояла проблема їхнього відновлення. У Європі поряд з цим постало питання європейської інтеграції як способу подолання ворожнечі й створення спільної ідентичності. 
"""

# Off the Shelf Model

model = TransformersNER("tner/roberta-large-wnut2017")

# output = model.predict([en_text])
visualize_entities(en_text, model)

# visualize_entities(uk_text, model)

# model_trained = TransformersNER("tner/roberta-large-wnut2017")
# print(model_trained.pipe_names)
#
# visualize_entities(uk_text, model_trained)