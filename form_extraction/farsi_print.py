import arabic_reshaper
from bidi.algorithm import get_display

# ...
farsi_char = [u"۰",u"۱",u"۲",u"۳",u"۴",u"۵",u"۶",u"۷",u"۸",u"۹",u"ا",u"ب",u"پ",u"ت",u"ث",u"ج",u"چ",u"ح",u"خ",u"د",u"ذ",u"ر",u"ز",u"ژ",u"س",u"ش",u"ص",u"ض",u"ط",u"ظ",u"ع",u"غ",u"ف",u"ق",u"ک",u"گ",u"ل",u"م",u"ن",u"و",u"ه",u"ی"]


for i in farsi_char:
    reshaped_text = arabic_reshaper.reshape(i)
    bidi_text = get_display(reshaped_text)
    print(bidi_text)



# reshaped_text = arabic_reshaper.reshape(u'ج')
# bidi_text = get_display(reshaped_text)
# print(bidi_text)
#pass_arabic_text_to_render(bidi_text)  # <-- This function does not really exist
# ...
