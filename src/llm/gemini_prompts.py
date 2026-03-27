"""
Sentiment Classification Prompts for Different Datasets
Each dataset has optimized instructions for better classification.
"""

FINANCE_PROMPT_INSTRUCTION = """
#SYSTEM ROLE
You are a highly precise Natural Language Processing (NLP) engine. Your task is to analyze a small subset of data provided in each request and accurately classify the sentiment of each individual item. Use your internal linguistic understanding to determine the most appropriate label based on the context of the text.

#DATA CONTEXT
Financial PhraseBank dataset. This dataset consists of 4,840 sentences from English-language financial news. The objective is to classify sentences based on their economic and financial sentiment, reflecting how the information might impact the perceived value or performance of the entities mentioned.

#CATEGORY DEFINITIONS
Assign exactly one of the following numerical labels to each entry:
0 -> 'negative'
1 -> 'neutral'
2 -> 'positive'

#FEW-SHOT EXAMPLES
[
  {
    "text": "In Q1 of 2009 , Bank of +àland 's net interest income weakened by 10 % to EUR 9.1 mn .",
    "label": 0
  },
  {
    "text": "Finnish Suominen Flexible Packaging is cutting 48 jobs in its unit in Tampere and two in Nastola , in Finland .",
    "label": 0
  },
  {
    "text": "Pretax profit decreased by 37 % to EUR 193.1 mn from EUR 305.6 mn .",
    "label": 0
  },
  {
    "text": "An estimated 30 pct of mobile calls are made from the home , and France Telecom hopes that 15 pct of its Orange clients will sign up for the service by the end of 2008 .",
    "label": 1
  },
  {
    "text": "The acquired business main asset is a mobile authentication and signing solution , branded as Tectia MobileID , which provides authentication to web e-mail , SSL-VPN , MS SharePoint , Tectia Secure Solutions and other applications and resources .",
    "label": 1
  },
  {
    "text": "Aug. 17 , 2010 ( Curbed delivered by Newstex ) -- And now , the latest from Racked , covering shopping and retail from the sidewalks up .",
    "label": 1
  },
  {
    "text": "Etteplan targets to employ at least 20 people in Borl+ñnge .",
    "label": 2
  },
  {
    "text": "Instead , Elcoteq has signed a non-binding Letter of Intent with another Asian strategic investor .",
    "label": 2
  },
  {
    "text": "Operating profit was EUR 139.7 mn , up 23 % from EUR 113.8 mn .",
    "label": 2
  }
]
#ENTRIES TO CLASSIFY
Below are the specific data items you need to analyze in this request:
"""

PROMPTS = {
    "imdb": {
        "name": "IMDb Movie Reviews",
        "labels": ["negative", "positive"],
        "instruction": """
#SYSTEM ROLE
You are a highly precise Natural Language Processing (NLP) engine. Your task is to analyze a small subset of data provided in each request and accurately classify the sentiment of each individual item. Use your internal linguistic understanding to determine the most appropriate label based on the context of the text.

#DATA CONTEXT
IMDB dataset having 50K movie reviews for natural language processing or Text analytics.
This is a dataset for binary sentiment classification.

#CATEGORY DEFINITIONS
Assign exactly one of the following numerical labels to each entry:
0 - negative
1 - positive

#FEW-SHOT EXAMPLES
[
  {
    "text": "There is no reason to see this movie. A good plot idea is handled very badly. In the middle of the movie everything changes and from there on nothing makes much sense. The reason for the killings are not made clear. The acting is awful. Nick Stahl obviously needs a better director. He was excellent in In the Bedroom, but here he is terrible. Amber Benson from Buffy, has to change her character someday. Even those of you who enjoy gratuitous sex and violence will be disappointed. Even though the movie was 80 minutes, which is too short for a good movie (but too long for this one),there are no deleted scenes in the DVD which means they never bothered to fill in the missing parts to the characters.  Don't spend the time on this one.",
    "label": 0
  },
  {
    "text": "Dark and bleak sets, thrilling music that cuts through your spin like aknife (or razor) a perfect cast lead by Broadway greats Hearn and Lansbury. This is exciting theatre flawlessly transferred to the small screen. Sondheim is the most talented songwriter of our age and Todd is his Masterpiece, from the Brechtian opening ballad to the darkly humorous Act I finale- \"A Little Priest\" where Lovett and Todd fantasize about the victims that will wind up in their meat pies , this play never ceases to thrill,excite and satisfy. Betsy Joslyn also excels as Johanna, even she, as the plays ingenue seems slightly mad.Edmund Lyndeck turns in a bravado performance as the corrupt Judge who lusts after Joslyn and is the subject of Todd's vendetta. Lansbury and Hearn command the show as only two great actor/stars can do. Other musical highlights include Todd's \"johanna\" Lovett's \"worst pies in London\" and the Act II opening 'GOD THATS GOOD\", And that is a title to describe this production !",
    "label": 1
  },
  {
    "text": "This movie is poorly written, hard-to-follow, and features bad performances and dialog from leads Jason Patric and Jennifer Jason Leigh. The premise, believable but weak (undercover narcotics agent succumbs to the drug underworld) deserved better than this Lili Fini Zanuck flop. The competent supporting cast (Sam Elliott, William Sadler, others) was not enough to save this film.  In addition, this movie also contains the absolute worst \"love\" scene in cinema.  Moreover, the soundtrack is vastly overrated; specifically the revolting, sappy-without-substance \"Tears in Heaven\" by the otherwise legendary Eric Clapton.  \"Rush\" is wholly unenjoyable from beginning to end.  2 of 10",
    "label": 0
  },
  {
    "text": "I thoroughly enjoyed this film for its humor and pathos. I especially like the way the characters welcomed Gina's various suitors. With friends (and family) like these anyone would feel nurtured and loved. I found the writing witty and natural and the actors made the material come alive.",
    "label": 1
  }
]

#ENTRIES TO CLASSIFY
Below are the specific data items you need to analyze in this request:

"""
    },
    
    "twitter": {
        "name": "Twitter Sentiment",
        "labels": ["irrelevant", "negative", "neutral", "positive"],
        "instruction": """
#SYSTEM ROLE
You are a highly precise Natural Language Processing (NLP) engine. Your task is to analyze a small subset of data provided in each request and accurately classify the sentiment of each individual item. Use your internal linguistic understanding to determine the most appropriate label based on the context of the text.

#DATA CONTEXT
This is an entity-level sentiment analysis dataset of twitter. Given a message the task is to judge the sentiment of the message about the entity. There are three classes in this dataset: Positive, Negative and Neutral. We regard messages that are not relevant to the entity (i.e. Irrelevant) as Neutral.

#CATEGORY DEFINITIONS
Assign exactly one of the following numerical labels to each entry:
0 - negative
1 - positive

#FEW-SHOT EXAMPLES
[
  {
    "text": "There is no reason to see this movie. A good plot idea is handled very badly. In the middle of the movie everything changes and from there on nothing makes much sense. The reason for the killings are not made clear. The acting is awful. Nick Stahl obviously needs a better director. He was excellent in In the Bedroom, but here he is terrible. Amber Benson from Buffy, has to change her character someday. Even those of you who enjoy gratuitous sex and violence will be disappointed. Even though the movie was 80 minutes, which is too short for a good movie (but too long for this one),there are no deleted scenes in the DVD which means they never bothered to fill in the missing parts to the characters.  Don't spend the time on this one.",
    "label": 0
  },
  {
    "text": "Dark and bleak sets, thrilling music that cuts through your spin like aknife (or razor) a perfect cast lead by Broadway greats Hearn and Lansbury. This is exciting theatre flawlessly transferred to the small screen. Sondheim is the most talented songwriter of our age and Todd is his Masterpiece, from the Brechtian opening ballad to the darkly humorous Act I finale- \"A Little Priest\" where Lovett and Todd fantasize about the victims that will wind up in their meat pies , this play never ceases to thrill,excite and satisfy. Betsy Joslyn also excels as Johanna, even she, as the plays ingenue seems slightly mad.Edmund Lyndeck turns in a bravado performance as the corrupt Judge who lusts after Joslyn and is the subject of Todd's vendetta. Lansbury and Hearn command the show as only two great actor/stars can do. Other musical highlights include Todd's \"johanna\" Lovett's \"worst pies in London\" and the Act II opening 'GOD THATS GOOD\", And that is a title to describe this production !",
    "label": 1
  },
  {
    "text": "CitizenX(1995) is the developing world's answer to Silence of the Lambs. Where `Silence' terrorized our peace of mind, `Citizen' exhausts and saddens us instead. This dramatization of the Chikatilo case translates rather well, thanks to a Westernized friendship between two Rostov cops who become equals.  CitizenX may also argue against(!) the death penalty far better than Kevin Spacey's The Life of David Gayle(2002).  Humans are Machiavellian mammals, under which lie limbic brains (lizard-logic). Why did two kids, who knew better, stone to death a toddler they kidnapped? Why do bloodthirsty women yell `li-lilililililii' at acts of OBSCENE terrorism? -My own term for this is `limbic domination', the lizard-logic urge to dominate an `enemy'. If you have the words `enemy'/`vengeance' in your vocabulary, you're easily capable of `limbic domination'.  In WWII-devastated 1980s Rostov (located at the mouth of the Don river near the Black Sea), nothing suppressed Andrei Chikatilo's urge for `limbic domination' from overpowering his layers of civilization. Chikatilo(Jeffrey DeMunn)'s easy victims were paupers, usually children, who rode the interurban train for fun, since they couldn't afford anything else.  CitizenX reminds us that the denials of a rampant Soviet bureaucracy cost the lives of 52 such `lambs'. Rostov's serial killer roamed free for almost 7 years AFTER the police arrested and let him go.  The politicization of crimefighting is harmful to police forces everywhere. Although policing routinely suffers from corruption all over the world, in the west, vote-grabbing by politicians can set up chronic inter-agency rivalries, stymieing a more coordinated response to crime. In the Soviet Union of CitizenX, however, Viktor Burakov(Stephen Rea)'s Killer Department was suffering from a repressive bureaucracy.  Geoffrey DeMunn plays the psychosexually inadequate Chikatilo with faultless but understated authority--to the point of complete obscurity. In real life, too, Chikatilo had a lifetime's experience blending in and evading capture.  His pursuer, on the other hand, sticks out as a strange bird, given to unheralded, naive outbursts. Perhaps by design, Stephen Rea gives a very strange performance as forensics chief Burakov. Rea's Russian accent is impenetrable; and his Burakov is humourless and sullen, at odds with everyone.  So it's Donald Sutherland who walks away with the picture. Sutherland's Col.Fetisov, Burakov's boss, and at first his only supporter, is an overly restrained, patient Militiaman whose dignified carriage bears testimony to decades of bureaucratic machinations. His reawakening as a logic-driven yet still passionate cop becomes the film's cornerstone idealism.  Joss Ackland does another turn as a vicious apparatchik, Secretary of Communist Ideology Bondarchuk, overseeing the investigation. Naturally, he quashed the arrest of the most likely suspect, a Communist, in 1984, a man carrying rope and a knife in his bag, supposedly going home: Andrei Chikatilo.  Soon, he replaced Burakov with another Moscow apparatchik, Detective Gorbunov(John Wood), insisting that the investigation now focus on `known homosexuals'. The funniest scene of this sad, sad film comes during Bondarchuk's & Gorbunov's institutionalized harassment: one stupid cop earnestly reports, `As I suspected, comrade, it's fornication. I've made some drawings'--cue howling laughter.  5yrs after the bodies began piling up, in 1987, the police finally tried soliciting criminal profiles. The only cooperating Soviet psychiatrist was Dr Aleksandr Bukhanovsky(Max Von Sydow), who termed the UNSUB `CitizenX'. He later also observed to Fetisov & Burakov that `...together you make a wonderful person'. We concur.  The drawn-out pace, spread over a decade, perfectly captures the institutional inertia of Glasnost--`openness'--that wasn't. The contrast with Perestroika--`restructuring'--couldn't've been greater for the case. Although Chikatilo was still prowling railway stations, police plans were about to bear fruit.  In 1990, Col.Fetisov was expeditiously promoted to General. His nemesis Bondarchuk disappeared off the scene, allowing the investigation to finally proceed without political interference. Staff, communications, publicity--suddenly all were available. In just one night of telephoning around, Fetisov got his depressed forensics chief access to the FBI's Serial Murder Task Force at Quantico, where, Fetisov discovered, staff are regularly rotated off serial murder cases to stave off just such psychological damage to investigators.  Fetisov advises his newly promoted forensics chief, now `Colonel' Burakov, of all these changes in an avalanche of confession that becomes the movie's powerhouse watershed scene. Fetisov's is the most tender apology I've ever seen on film: `Privately, I offer my deepest apologies to you and your wife. I hope that someday you can forgive me my ignorance', he almost whispers.  A HBO production, CitizenX is a film of the highest caliber. Not only do the exteriors look authentically bleak (shot exclusively in the most run-down parts of otherwise spectacular Budapest), but Randy Edelman's soaring soundtrack is entirely overwhelming--and frequently our only respite from the bleak brutality. Those who speak Hungarian will recognize the many Hungarian accents and credits.  Chikatilo's actual murders are depicted as bleak, aberrant behaviour born of character flaws and ignorance in an equally bleak world. This makes the murders seem not-entirely-out-of-place--but of course they were. As President Kennedy reminded us, `we all cherish the futures of our children'.  CitizenX communicates perfectly that killing is far more grisly and obscene than any vengeance fantasy might imply. Serial rapists rape to dominate; serial killers kill to dominate. So do some soldiers. Such `limbic dominators' make poor humans.  WARNING-SPOILER:----------------------------------------------- The real Andrei Chikatilo WAS the world's most prolific known serial killer. Convicted, he was executed in 1992 in the manner of all Soviet Union death sentences: one shot, in the back of the head. Foolishly, such methods destroy any possibility of studying a deviant brain after death.  Conclusion:------------------------------------------------------------ The best outcome is always the prevention of killings, not their prosecution. Executions merely guarantee society's failure to learn from the complex reality of victims' deaths when we dispatch even anecdotal evidence of HOW/WHY they died. Nor do killers learn regret if they're dead.  Vengeance doesn't unkill victims. Baying for the killer's blood constitutes nothing better than counter-domination--once it's too late.  Vengeance on behalf of the grieving isn't justice for the deceased--it's appeasement of the living.(10/10)",
    "label": 1
  },
  {
    "text": "This is one of the worst pieces of cinema I have seen in some time. This is also my first review so you can tell I must hate this film at lot.  Don't get me wrong, I like my serious films. I don't like Hollywood too much, I tend to like French, Italian, offbeat US or anything that tries to communicate something sensible.  But this was awful. Why? 1. The plot (such as it was) was entirely unbelievable, even though the director seems to be hinging everything on a feeling of realism.  2. The main character has nothing to recommend him. Does he smoke for coolness or to show us his angst? For goodness sake this guy is meant to be an ex-dodgy mafia lawyer. Are we meant to care more because that he is also one of the meanest unfriendly people you could ever meet? And he smokes...so he must have deep personal issues. Pop psychology at it's best. In the final moments, I almost cheered as he gets buried in the cement. Best place for him. And I thought that was about the only good scene. Or maybe that was because it was so close the final credits.  3. The entirely tired and unbelievable interest in the main character from the beautiful girl. She was there simply because beautiful girls always have odd sexual relationships with old, old men with a deep and meaningful personality (as demonstrated by smoking). Happens all the time. In really bad films that is.  4. The pace was so leaden. I like slow, I like careful. But this was just deathly.  5-50 a bunch of other stuff that I really can't be bothered to write.  Awful.",
    "label": 0
  },
  {
    "text": "This movie is poorly written, hard-to-follow, and features bad performances and dialog from leads Jason Patric and Jennifer Jason Leigh. The premise, believable but weak (undercover narcotics agent succumbs to the drug underworld) deserved better than this Lili Fini Zanuck flop. The competent supporting cast (Sam Elliott, William Sadler, others) was not enough to save this film.  In addition, this movie also contains the absolute worst \"love\" scene in cinema.  Moreover, the soundtrack is vastly overrated; specifically the revolting, sappy-without-substance \"Tears in Heaven\" by the otherwise legendary Eric Clapton.  \"Rush\" is wholly unenjoyable from beginning to end.  2 of 10",
    "label": 0
  },
  {
    "text": "I thoroughly enjoyed this film for its humor and pathos. I especially like the way the characters welcomed Gina's various suitors. With friends (and family) like these anyone would feel nurtured and loved. I found the writing witty and natural and the actors made the material come alive.",
    "label": 1
  }
]

#ENTRIES TO CLASSIFY
Below are the specific data items you need to analyze in this request:

"""
    },
    
    "finance_50": {
        "name": "Finance News (50% Agreement)",
        "labels": ["positive", "negative", "neutral"],
        "instruction": FINANCE_PROMPT_INSTRUCTION
    },
    
    "finance_66": {
        "name": "Finance News (66% Agreement)",
        "labels": ["positive", "negative", "neutral"],
        "instruction": FINANCE_PROMPT_INSTRUCTION
    },
    
    "finance_75": {
        "name": "Finance News (75% Agreement)",
        "labels": ["positive", "negative", "neutral"],
        "instruction": FINANCE_PROMPT_INSTRUCTION
    },
    
    "finance_all": {
        "name": "Finance News (100% Agreement)",
        "labels": ["positive", "negative", "neutral"],
        "instruction": FINANCE_PROMPT_INSTRUCTION
    },
    
    "yelp": {
        "name": "Yelp Reviews",
        "labels": ["positive", "negative", "neutral"],
        "instruction": """
#SYSTEM ROLE
You are a highly precise Natural Language Processing (NLP) engine. Your task is to analyze a small subset of data provided in each request and accurately classify the sentiment of each individual item. Use your internal linguistic understanding to determine the most appropriate label based on the context of the text.

#DATA CONTEXT
Yelp Reviews dataset. This dataset consists of reviews for various businesses, including restaurants, hotels, and services. The objective is to predict the star rating based on the text of the review, reflecting the customer's level of satisfaction and overall experience.

#CATEGORY DEFINITIONS
Labels corresponds to the score associated with the review (between 1 and 5).
1 - Extreme dissatisfaction (e.g., terrible service, health concerns, or complete failure of expectations).
2 - Dissatisfaction (e.g., significant flaws, overpriced for the quality, or poor experience despite some minor redeeming qualities).
3 - Neutral/Mixed (e.g., "okay" experience, average quality, or a mix of strong positives and strong negatives).
4 - High satisfaction (e.g., great experience, minor room for improvement, or very good value).
5 - Exceptional experience (e.g., perfect service, outstanding quality, or highly recommended).

#FEW-SHOT EXAMPLES
[
  {
    "text": "Do NOT schedule an event here!\\n\\nE2 cancelled my brother's rehearsal dinner at the last minute.  We were out of options at that point, and had to settle for place we could not afford.  The owner decided to give the space to his friend on that date, so they booted us out.  Very unprofessional. \\n\\nThey were not apologetic, either.  It took a scathing email (which I never write) and a few phone calls to have them respond.  And we have yet to receive our security deposit.  \\n\\nPlease do not make the same mistake my family did.  Look somewhere else for anything more than a small dinner reservation.",
    "label": 0
  },
  {
    "text": "Excited to go and try this place out, limited bar, not many flavored vodkas and food was mediocre at best, Maggiano's is much better, will not be back!",
    "label": 0
  },
  {
    "text": "McDonalds\\nShe Calls My Name\\nGolden Arches are to Blame\\nEntire Fest \\nCost the Least\\nPeople Flock\\nAround the Clock\\nService so Slow\\nWhy Do I Even Go\\nWaistline Busting With Regret\\nFailed the Supersize to Fret\\nEat Some Fries Here and There\\nSuddenly My Soul Flies Through the Air\\nMcDonalds Doth Forsaken Me\\nSupersize Diet Coke\\nOh, Whoopee...\\n\\n\\n\\nPro-tip: Go to the location on 7th Ave and Indian School instead! It will take you an extra 3 minutes to get there but the staff, service, and location are all much nicer. This location just opened up in early 2014. Very modern for a McDonalds.",
    "label": 1
  },
  {
    "text": "First time here and I ordered corned beef and cabbage. Now this is a \\\"irish\\\" theme. If Paddy McFinn ate this man he would be pissed off. There was more bacon mixed in the cabbage that there was cabbage. The potatoes taste like half cooked microwave spuds hacked into cubes and thrown into a pan for 3 seconds.\\n\\nThe corned beef was actually good. But for there \\\"signature\\\" dish, u couldnt f**k it any more.\\n\\nVery disappointing and expensive for what should be such an easy and cheap dish. (17.99 making it one of the most expensive dishes).\\n\\nAt 7pm there isnt any one in here. Gordon Ramsey wouldnt be able to save this place!!!",
    "label": 1
  },
  {
    "text": "Not really a fan of these types of places to begin with. Never had issues with service but it is really overpriced. I love the flat screen HD tv's. Food was okay, nothing impressive. Drinks are decent.",
    "label": 2
  },
  {
    "text": "Prices are reasonable.\\n\\nWhat I got did not match the description though.  The roll I ordered said \\\"very spicy\\\" and this wasn't spicy at all.  I did not like this weird white sauce poured over my rolls either.  \\nI did taste some of what my friend ordered and it was delicious.  I'll definitely be back, but I will be a bit more cautious of what I am being served.",
    "label": 2
  },
  {
    "text": "I love that this place is open late.  I have come in to do the Korean BBQ before and have come in to just have some spicy tofu soup by itself as well.  Staff is always pleasant and pretty attentive.  I have never had any issues and tend to eat here once, every time I go to Vegas.\\n\\nThere is a decent variety of meats to grill---and if you're feeling hungry, the all you can eat is the way to go.  I think the banchan varies greatly as it does at all other places, so these can be hit or miss.  The Korean BBQ gets pretty busy during the late nights so be prepared to wait----but since there is an all you can eat option, the wait is totally worth it.",
    "label": 3
  },
  {
    "text": "i normally really like this bank. every time i go in there is no one in there so i never have to wait. so as far as banks go i really like it its quick and easy to pay my bills.",
    "label": 3
  },
  {
    "text": "Great food, terrific service, reasonable prices. I definitely recommend.",
    "label": 4
  },
  {
    "text": "I wear those rimless, wire glasses and somehow broke the wire nose pieces off .  I have trifocals and so they didn't sit right for hands-free use of the bottom two lenses.    I dropped into Arties on my way to work and talked to a woman stocking the isles.  She smiled, took a close look at my glasses.  Then walked to the hose section:  cut a piece of clear plastic hose, slit it along the side and pushed it onto the nose-bridge wire.  It works very well, I don't look to odd and it will last until my new glasses arrive!! Thanks artie and staff for your hero magic, you saved the day!",
    "label": 4
  }
]

#ENTRIES TO CLASSIFY
Below are the specific data items you need to analyze in this request:


"""
    },
    
    "tweet_sentiment": {
        "name": "Tweet Eval - Sentiment",
        "labels": ["negative", "neutral", "positive"],
        "instruction": """
#SYSTEM ROLE
You are a highly precise Natural Language Processing (NLP) engine. Your task is to analyze a small subset of data provided in each request and accurately classify the sentiment of each individual item. Use your internal linguistic understanding to determine the most appropriate label based on the context of the text.

#DATA CONTEXT
Twitter sentiment dataset. This dataset contains short, informal tweets from various users. The objective is to identify the overall emotional tone of each tweet, taking into account the use of hashtags, mentions, and common internet slang.

#CATEGORY DEFINITIONS
Assign exactly one of the following numerical labels to each entry:
0 - Negative
1 - Neutral
2 - Positive

#FEW-SHOT EXAMPLES
[
  {
    "text": "\"#CNET - Just bought my 1st iPad, iPad3, feeling real burned, mad, about iPad4 so soon. Grrr. REALLY mad! Don't even care about mini now,\"",
    "label": 0
  },
  {
    "text": "\"Mother's Day is coming like tomorrow or is it today? idk, but i didn't prepare anything for me's mom.\"",
    "label": 0
  },
  {
    "text": "@user @user @user  I think this is the motive of the Yakub's laywers for pursuing the case",
    "label": 1
  },
  {
    "text": "@user From this performance it is obvious why David Miliband comes up 1st when you google Miliband! #onetowatch",
    "label": 1
  },
  {
    "text": "Kanye West was honored in a big way during Sunday night's MTV Video Music Awards by receiving the Michael Jackso...",
    "label": 2
  },
  {
    "text": "Have you heard about Amazon Prime day? Don't miss it - July 15th! Yay! #ad #PrepForPrime @user #HappyPrimeDay",
    "label": 2
  }
]


ENTRIES TO CLASSIFY
Below are the specific data items you need to analyze in this request:


"""
    },
    
    "tweet_irony": {
        "name": "Tweet Eval - Irony",
        "labels": ["non_irony", "irony"],
        "instruction": """
#SYSTEM ROLE
You are a highly precise Natural Language Processing (NLP) engine. Your task is to analyze a small subset of data provided in each request and accurately classify the sentiment of each individual item. Use your internal linguistic understanding to determine the most appropriate label based on the context of the text.

#DATA CONTEXT
Twitter irony detection dataset. This dataset contains short, informal tweets where the goal is to distinguish between literal language and irony (including sarcasm, satire, or rhetorical devices where the intended meaning differs from the literal one).

#CATEGORY DEFINITIONS
Assign exactly one of the following numerical labels to each entry:
0 - non_irony
1 - irony

#FEW-SHOT EXAMPLES
[
  {
    "text": "@user You truly are my son.",
    "label": 0
  },
  {
    "text": "Favorite @user if you love Romance Novels. #Love #Drama #Romance",
    "label": 0
  },
  {
    "text": "@user thanks for leaving the dish for us  #badbusiness",
    "label": 1
  },
  {
    "text": "@user @user Spoiled, rancid food AND guns in the aisles? Rats in @user great.  #GroceriesNotGuns",
    "label": 1
  }
]


ENTRIES TO CLASSIFY
Below are the specific data items you need to analyze in this request:

"""
    },
}

def get_prompt_for_dataset(dataset_key: str) -> dict:
    if dataset_key not in PROMPTS:
        raise ValueError(f"No prompt defined for dataset: {dataset_key}")
    return PROMPTS[dataset_key]

def get_instruction(dataset_key: str) -> str:
    return get_prompt_for_dataset(dataset_key)["instruction"]

# def get_labels(dataset_key: str) -> list[str]:
#     return get_prompt_for_dataset(dataset_key)["labels"]

def get_dataset_name(dataset_key: str) -> str:
    return get_prompt_for_dataset(dataset_key)["name"]