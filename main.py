import textwrap
wrapper = textwrap.TextWrapper(width=70)

import summarizer

summarizer.initialize()

examples = [
    "It was a sunny day when I went to the market to buy some flowers. But I only found roses, not tulips.",
    "It’s the posing craze sweeping the U.S. after being brought to fame by skier Lindsey Vonn, soccer star Omar Cummings, baseball player Albert Pujols - and even Republican politician Rick Perry. But now four students at Riverhead High School on Long Island, New York, have been suspended for dropping to a knee and taking up a prayer pose to mimic Denver Broncos quarterback Tim Tebow. Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were all suspended for one day because the ‘Tebowing’ craze was blocking the hallway and presenting a safety hazard to students. Scroll down for video. Banned: Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll (all pictured left) were all suspended for one day by Riverhead High School on Long Island, New York, for their tribute to Broncos quarterback Tim Tebow. Issue: Four of the pupils were suspended for one day because they allegedly did not heed to warnings that the 'Tebowing' craze at the school was blocking the hallway and presenting a safety hazard to students.",
    "Hamas released a second group of Israeli and foreign hostages on Saturday night in exchange for the release of Palestinian prisoners, the Israeli authorities said Sunday morning, after an hourslong delay raised fears that a fragile truce in Gaza could collapse altogether. Qatar, which helped broker the deal alongside Egypt, said that two mediators had managed to overcome an impasse between Israel and Hamas. In the end, Israel confirmed that Hamas had handed 13 Israelis — eight children and five women — to the International Committee of the Red Cross in Gaza. They were taken in a convoy across the Rafah crossing to Egypt, then transported to Israel, where they were delivered to hospitals, the Israeli authorities said. Four Thai nationals were also released. Within hours, 39 Palestinian prisoners were released by Israel, Israel’s prison service said early on Sunday. There was a similar swap on Friday. The prisoners affairs commission of the Palestinian Authority confirmed that Red Cross buses with detainees had left Ofer prison, outside the West Bank city of Ramallah, to take them to Al-Bireh Municipality. The resumption of the deal late Saturday came after a tense day in which it appeared the fragile temporary cease-fire agreement might crumble. Hamas had threatened to postpone the second hostages-for-prisoners trade, claiming Israel had reneged on parts of the agreement. The armed group, which controls Gaza, said Israel had not allowed enough aid to reach northern Gaza and had not released Palestinian prisoners according to agreed-upon terms.",
    "The stabbing on Friday of Derek Chauvin, the former Minneapolis police officer convicted of murdering George Floyd in 2020, at a special unit inside a Tucson, Ariz., prison is the latest in a series of attacks against high-profile inmates in the troubled, short-staffed federal Bureau of Prisons. The assault comes less than five months after Larry Nassar, the doctor convicted of sexually abusing young female gymnasts, was stabbed multiple times at the federal prison in Florida. It also follows the release of Justice Department reports detailing incompetence and mismanagement at federal detention centers that led to the deaths in recent years of James Bulger, the Boston gangster known as Whitey, and Jeffrey Epstein, who had been charged with sex trafficking. The Federal Bureau of Prisons confirmed that an inmate at the Tucson prison was stabbed around 12:30 p.m. on Friday, though the bureau did not identify Mr. Chauvin, 47, by name. The agency said in a statement that the inmate required “life-saving measures” before being rushed to a hospital emergency room nearby. The office of Keith Ellison, the Minnesota attorney general who prosecuted the former police officer, identified the inmate as Mr. Chauvin. He is likely to survive, according to two people with knowledge of the situation who were not authorized to discuss the incident publicly. On Saturday, the prison remained on lockdown while law enforcement agencies, including the Federal Bureau of Investigation, examined the crime scene and interviewed witnesses. Family visits to the facility have been suspended indefinitely, according to the prison’s website. The facility in Tucson where Mr. Chauvin was stabbed is referred to as a “dropout yard,” one of several special protective units within the Federal Bureau of Prisons system housing informants, people convicted of sex crimes, former gang members and former law enforcement personnel, among others, according to Joe Rojas, who retired earlier this month as president of the union local representing workers at the Federal Correctional Complex near Coleman, Fla.",
]

for sentence in examples: 
    print(wrapper.fill(sentence), '\n')
    print(summarizer.summarize(sentence, mode='mbr'), '\n')