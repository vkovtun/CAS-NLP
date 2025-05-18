from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
_SENT_RE = re.compile(r"[^.!?]+[.!?]?")
_WS = set(" \t\n\r\f\v")

ANSI_BOLD = "\033[1m"
ANSI_BLUE = "\033[94m"
ANSI_RESET = "\033[0m"


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Return trimmed *(start, end)* indices for every sentence in *text*."""
    spans: List[Tuple[int, int]] = []
    for m in _SENT_RE.finditer(text):
        s, e = m.start(), m.end()
        while s < e and text[s] in _WS:  # strip leading whitespace
            s += 1
        spans.append((s, e))
    return spans


def _render(text: str, ent_spans: Iterable[Tuple[int, int, str]]) -> str:
    """Return *text* with colourised entity tags (ANSI)."""
    out, last = [], 0
    for s, e, lbl in ent_spans:
        out.append(text[last:s])
        out.append(f"{ANSI_BOLD}{text[s:e]}{ANSI_RESET}[{ANSI_BLUE}{lbl}{ANSI_RESET}]")
        last = e
    out.append(text[last:])
    return "".join(out)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def visualize_entities(text: str, ner_pipe, *, min_prob: float | None = None) -> None:
    """Print *text* with entities highlighted, using a 🤗 pipeline.

    Parameters
    ----------
    text : str
        Raw input text.
    ner_pipe : transformers.pipelines.Pipeline
        A *token-classification* pipeline (e.g. from :pyfunc:`transformers.pipeline`).
    min_prob : float, optional
        Only show entities whose confidence *score* ≥ *min_prob*.
    """
    sent_spans = _sentence_spans(text)
    ent_spans: List[Tuple[int, int, str, float]] = []  # start, end, label, score

    for (sent_start, sent_end) in sent_spans:
        sentence = text[sent_start:sent_end]
        for res in ner_pipe(sentence):
            score = float(res["score"])
            if min_prob is not None and score < min_prob:
                continue
            label = res.get("entity_group", res)
            start = sent_start + int(res["start"])
            end = sent_start + int(res["end"])
            ent_spans.append((start, end, label, score))

    ent_spans.sort(key=lambda t: t[0])
    print(_render(text, ((s, e, l) for s, e, l, _ in ent_spans)))


def main():
    uk_text = """
    Друга світова війна — глобальний збройний конфлікт, що тривав від 1 вересня 1939 року до 2 вересня 1945 року. У війні взяло участь понад 60 країн, зокрема всі великі держави, які утворили два протилежні військові табори: блок країн Осі та антигітлерівську коаліцію («союзники»). Безпосередню участь у бойових діях брали понад 100 мільйонів осіб. Супротивні держави кинули всі економічні, промислові та наукові можливості на потреби фронту, стираючи різницю між цивільними та військовими ресурсами. Загальні людські втрати коливаються між 50 й 80 мільйонами осіб, більшість із яких були мешканцями Радянського Союзу та Китаю. Друга світова війна відзначилася численними масовими вбивствами і злочинами проти людяності, насамперед Голокостом, стратегічними килимовими бомбардуваннями та єдиним в історії військовим застосуванням ядерної зброї.
    Основними причинами війни стали політичні суперечності, породжені недосконалою Версальською системою, та агресивна експансіоністська політика нацистської Німеччини, Японської імперії та Італії. 1 вересня 1939 року гітлерівські війська вторглися в Польщу. 3 вересня Велика Британія та Франція оголосили Німеччині війну. Упродовж 1939—1941 років завдяки серії успішних військових кампаній та низки дипломатичних заходів Німеччина захопила більшу частину континентальної Європи. Саме тоді й Радянський Союз анексував (повністю або частково) території сусідніх європейських держав: Польщі, Румунії, Фінляндії та країн Балтії, що відійшли до його сфери впливу на підставі Пакту Молотова — Ріббентропа. Після початку бойових дій у Північній Африці та падіння Франції в середині 1940 року війна тривала насамперед між країнами Осі та Великою Британією, повітряні сили якої зуміли відбити німецькі повітряні атаки. У цей же час бойові дії поширились на Балканський півострів та Атлантичний океан. Японія окупувала частину Китаю та Південно-Східної Азії, взявши під контроль важливі джерела сировини.
    22 червня 1941 року війська країн Осі чисельністю 3.5 мільйонів осіб вторглися в Радянський Союз, маючи на меті завоювання «життєвого простору» в Східній Європі. Відкривши найбільший в історії сухопутний фронт, німецькі війська спершу доволі швидко окупували західні регіони СРСР, однак в битві за Москву зазнали поразки. В цей же час Японія віроломно напала на США та підкорила західну частину Тихого океану. Задля протистояння агресії країн Осі створено Антигітлерівську коаліцію 26 країн, в окупованих країнах розгорнувся рух опору. У лютому 1943 радянська армія здобула перемогу під Сталінградом. У Північній Африці німецькі та італійські війська зазнали поразки під Ель-Аламейном. Просування Японії зупинили сили американців і австралійців у битві за Мідвей. У 1943 році після низки військових невдач Гітлера на Східному фронті, висадки союзників у Сицилії та Італії, що призвело до капітуляції останньої, і перемог США на Тихому океані, країни Осі втратили ініціативу та перейшли до стратегічного відступу на всіх фронтах. У 1944 році армії західних альянтів визволили Західну та Центральну Європу, у той час як радянські війська вигнали війська Німеччини та окупантів з власної території та країн Східної й Південно-Східної Європи.
    Протягом 1944 та 1945 років Японія зазнала великих втрат у материковій Азії, у Південному Китаї та Бірмі; союзники знищили японський флот і заволоділи ключовими островами в західній частині Тихого океану. Німеччина опинилася в щільному кільці. До кінця квітня 1945 року радянські війська заволоділи значною частиною її території, зокрема й Берліном; Адольф Гітлер вчинив самогубство. 8 травня керівництво Вермахту підписало Акт про беззастережну капітуляцію. Ця дата вважається Днем перемоги над нацизмом в Європі. Після опублікування 26 липня 1945 Потсдамської декларації та відмови Японії капітулювати на її умовах США скинули атомні бомби на міста Хіросіму і Нагасакі 6 і 9 серпня відповідно. У серпні 1945 Радянський Союз розгорнув бойові дії проти Японії. Неминуче вторгнення американців на японський архіпелаг, а також можливість інших атомних бомбардувань змусили керівництво цієї острівної країни здатися. Акт про капітуляцію Японії підписали 2 вересня 1945 року на борту лінкора «Міссурі». Війна в Азії закінчилась, закріпивши загальну перемогу Антигітлерівської коаліції.
    Друга світова стала наймасштабнішою та найкривавішою війною в історії людства, великим переламом XX століття, що докорінно змінив політичну карту і соціальну структуру світу. Для сприяння розвитку міжнародного співробітництва та запобігання майбутнім конфліктам створено Організацію Об'єднаних Націй. Післявоєнний порядок утвердив гегемонію Сполучених Штатів і Радянського Союзу, суперництво яких призвело до утворення капіталістичного й соціалістичного таборів та початку Холодної війни. Світовий вплив європейських держав значно ослаб, почався процес деколонізації Азії та Африки. Перед країнами, чиї галузі економіки були знищені, гостро стояла проблема їхнього відновлення. У Європі поряд з цим постало питання європейської інтеграції як способу подолання ворожнечі й створення спільної ідентичності.
    """.strip()

    cs_text = """
    Druhá světová válka byl globální vojenský konflikt v letech 1939–1945, jehož se zúčastnila většina států světa. Tento konflikt se stal s více než 62 miliony obětí na životech dosud největším a nejvíce zničujícím válečným střetnutím v dějinách lidstva.
    Brzy po okupaci zbytku Československa 15. března 1939 vypukla válka v Evropě. Začala dne 1. září 1939, když nacistické Německo přepadlo Polsko. Krátce poté vyhlásily Francie, Spojené království a státy Commonwealthu Německu válku. 17. září napadl Polsko i Sovětský svaz (SSSR). Německé invazi do Polska předcházela jeho smlouva o neútočení se SSSR, takzvaný pakt Ribbentrop–Molotov, podepsaná 23. srpna 1939. V tajném protokolu k tomuto paktu si tyto dva státy dočasně rozdělily sféry vlivu tzv. demarkační Curzonovu linii. Byla vytyčena v roce 1919 mezi druhou polskou republikou a Sovětským svazem, dvěma novými státy, které vznikly po první světové válce. Curzonova linie vznikla jako diplomatický podklad pro budoucí dohodu o hranicích. Vytyčila ji Rada Dohody podle usnesení z 8. prosince 1919. V důsledku toho byl de facto uvolněn prostor pro vojenskou expanzi obou stran. Ze strany Německa byl na části území Polska až k demarkační linii vytvořen Generalgouvernement – jako správní jednotka utvořená 26.10.1939 na základě Hitlerova dekretu, která zahrnovala část okupovaného území původního meziválečného Polska, které nebylo začleněno do Třetí říše (vojvodství Kielecké, Krakovské, Lublinské, část Lodžského a Varšavského). Sídlem správy byl Krakov. V čele Generálního gouvernementu stál válečný zločinec Hans Frank, pod jehož vedením docházelo k brutálnímu útisku a cílenému vyhlazování nejen polského, ale i židovského obyvatelstva v Polsku.
    Blesková válka na západě Evropy začala dne 10. května 1940, kdy německý Wehrmacht na rozkaz Adolfa Hitlera překročil hranice Belgie, Nizozemska a Lucemburska, a poté obešel obrannou Maginotovu linii. Po rychlé porážce francouzské armády vstoupila německá vojska 14. června do Paříže. Francie kapitulovala 22. června 1940 a do listopadu 1942 německá armáda postupně obsadila celou zemi.
    Sovětský svaz se stal jedním ze Spojenců druhé světové války proti Ose Berlín–Řím–Tokio poté, co nacistické Německo zahájilo dne 22. června 1941 proti němu rozsáhlou a ničivou operaci Barbarossa. Zprvu musela Rudá armáda ustupovat až k Moskvě. Po urputných bojích, např. v bitvě u Stalingradu (podzim 1942 – zima 1943), o Kurský oblouk, či obležení Leningradu (dnešní Petrohrad) atd., začala sovětská vojska zatlačovat Wehrmacht západním směrem a dne 2. května 1945 dobyla Berlín.
    Dne 7. července 1937 se udál incident na mostě Marca Pola v Pekingu. Tímto relativně malým vojenským střetnutím de facto vznikla druhá čínsko-japonská válka, zprvu bez formálního válečného stavu mezi Japonským císařstvím a Čínskou republikou. Japonsko poté pokračovalo ve své rozsáhlé expanzi proti čínským územím, pak přepadlo a dobylo řadu zemí v jihovýchodní Asii. Dne 7. prosince 1941 zaútočila letadla z japonských letadlových lodí na americkou námořní základnu Pearl Harbor na Havajských ostrovech. Den poté vstoupily Spojené státy americké do války proti Japonsku. Teprve 9. prosince 1941 vyhlásila čínská vláda oficiálně Japonsku válku. 11. prosince 1941 vyhlásily nacistické Německo a Itálie válku Spojeným státům, čímž byl utvrzen stav globálního konfliktu.
    Konec války v Evropě nastal 8. května 1945 bezpodmínečnou kapitulací Německa. Po americkém svržení atomových bomb ve dnech 6. a 9. srpna 1945 na města Hirošima a Nagasaki kapitulovalo Japonsko 2. září 1945.
    Příčiny války bývají hledány v důsledcích ideologií a politických směrů, jako jsou nacionalismus a imperialismus. Podle některých historiků byla jednou z hlavních příčin nespokojenost vládnoucích kruhů Německa s dopady Versailleské smlouvy, která měla prohloubit pocit ponížení po prohrané první světové válce, a v následcích velké hospodářské krize na přelomu dvacátých a třicátých let. Tyto vlivy zásadním způsobem oslabily mnoho evropských států, čímž umožnily vzestup nacismu a italského fašismu.
    Druhou světovou válku provázely v dosud nevídané míře válečné zločiny, zločiny proti lidskosti a nehumánní zacházení s válečnými zajatci, zvláště se sovětskými vojáky ze strany Německa. Průběhem bojů bylo podstatně zasaženo rovněž civilní obyvatelstvo, jež utrpělo obrovské ztráty. Nejhorším příkladem genocidy se stal holokaust (šoa), kterému na základě nacistické rasové ideologie padlo za oběť šest milionů Židů v koncentračních táborech a na jiných místech v mnoha zemích Evropy. Masakr čínského obyvatelstva, který spáchali Japonci v Nankingu v prosinci 1937, byl jedním z největších zločinů. V rámci operace Intelligenzaktion v roce 1939 zavraždili němečtí nacisté 60 000 až 100 000 příslušníků polské inteligence, důstojníků a představitelů měst i státu. V roce 1940 provedla sovětská NKVD tzv. katyňský masakr, při kterém bylo povražděno přibližně 22 000 polských důstojníků a příslušníků inteligence. Milionové oběti utrpělo slovanské civilní obyvatelstvo – Rusové, Ukrajinci, Bělorusové, Poláci a jiní – na územích východní fronty, kde ztratilo životy osm milionů lidí. Ti podléhali nemocem a hladu vyvolaným válečnými operacemi a masakrům páchaným na územích obsazených Wehrmachtem a jednotkami Waffen-SS.
    Válečné úsilí pohlcovalo téměř veškerý lidský, ekonomický, průmyslový a vědecký potenciál všech zúčastněných národů. Mnoho států utrpělo nepředstavitelné materiální ztráty a devastaci svého kulturního dědictví. Lze proto hovořit o tom, že se jednalo o totální válku. Téměř všechny zúčastněné strany se v menší či větší míře odchylovaly od požadavku vést válku „civilizovanými metodami“. I když Spojené království v roce 1940 odmítalo plošné nálety na nepřátelská města, posléze se k nim spolu se Spojenými státy samo uchýlilo.
    V samotném závěru světové války byla ustavena Organizace spojených národů, jejímž ústředním cílem byla a je i v současnosti prevence vzniku dalších válečných konfliktů. Po skončení války upevnily vítězné mocnosti USA a SSSR své postavení dvou světových supervelmocí. Jejich stále větší vzájemný antagonismus vedl k bipolárnímu rozdělení světa a k počátku studené války. První generální tajemník Komunistické strany Sovětského svazu Josif Stalin spustil napříč evropským kontinentem tzv. železnou oponu, která od sebe oddělila západní svět a státy ve Východní Evropě, především z obavy vojenského obsazení zemí Východní Evropy. Vedlejším efektem války byl také vzrůst požadavků na právo na sebeurčení mezi národy ovládanými koloniálními mocnostmi, což vedlo k akceleraci dekolonizačních hnutí v Asii a v Africe.
    Jednalo se o nejrozsáhlejší válku v dějinách, které se přímo účastnilo více než 100 milionů lidí z více než 30 zemí. Ve stavu totální války vrhli hlavní účastníci do válečného úsilí veškeré své hospodářské, průmyslové a vědecké kapacity, čímž se smazaly rozdíly mezi civilním a vojenským obyvatelstvem.
    """.strip()

    cs_text = """
    Druhá světová válka byl globální vojenský konflikt v letech 1939–1945, jehož se zúčastnila většina států světa. Tento konflikt se stal s více než 62 miliony obětí na životech dosud největším a nejvíce zničujícím válečným střetnutím v dějinách lidstva.
    """.strip()

    # Build pipeline once; reuse for many calls
    tokenizer = AutoTokenizer.from_pretrained("ivlcic/xlmr-ner-slavic")
    model = AutoModelForTokenClassification.from_pretrained("ivlcic/xlmr-ner-slavic")
    ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    visualize_entities(cs_text, ner)
    # visualize_entities(uk_text, ner)


if __name__ == "__main__":
    main()
