# Paraphrase review

Check that each paraphrase applies to exactly the same cases as the original. Flag any added, dropped, or changed condition.

## mfrc

### Care (`care`)

- **Original:** Intuitions about avoiding emotional and physical damage or harm to others. It underlies virtues of kindness, gentleness, and nurturing, and vices of meanness, violence, and abuse.
- **p0:** Gut-level concerns about keeping others from suffering emotional or physical damage or harm. It supports the virtues of kindness, gentleness, and nurturing, and the vices of meanness, violence, and abuse.
- **p1:** The virtues of kindness, gentleness, and nurturing, and the vices of meanness, violence, and abuse, rest on intuitions about not inflicting emotional or physical damage or harm on others.

### Equality (`equality`)

- **Original:** Intuitions about egalitarian treatment and equal outcome for all individuals and groups.
- **p0:** Gut-level concerns that all individuals and groups be treated in an egalitarian way and receive equal outcomes.
- **p1:** Intuitions favoring equal outcomes for every individual and group, and egalitarian treatment of them.

### Proportionality (`proportionality`)

- **Original:** Intuitions about individuals getting rewarded in proportion to their merit (i.e., effort, talent, or input).
- **p0:** Gut-level concerns that individuals be rewarded according to their merit, meaning their effort, talent, or input.
- **p1:** Intuitions that the reward an individual receives should match their merit (effort, talent, or input).

### Loyalty (`loyalty`)

- **Original:** Intuitions about in-group cooperation and out-group competition. It underlies virtues of patriotism and self-sacrifice for the group.
- **p0:** Gut-level concerns about cooperating within the in-group and competing against out-groups. It supports the virtues of patriotism and self-sacrifice for the group.
- **p1:** The virtues of patriotism and self-sacrifice for the group rest on intuitions about in-group cooperation and out-group competition.

### Authority (`authority`)

- **Original:** Intuitions about deference toward legitimate authorities and high-status individuals. It underlies virtues of leadership and respect for tradition.
- **p0:** Gut-level concerns about showing deference to legitimate authorities and high-status individuals. It supports the virtues of leadership and respect for tradition.
- **p1:** The virtues of leadership and respect for tradition rest on intuitions about deferring to legitimate authorities and high-status individuals.

### Purity (`purity`)

- **Original:** Intuitions about avoiding bodily and spiritual contamination and degradation. It underlies virtues of sanctity, nobility, and cleanliness.
- **p0:** Gut-level concerns about steering clear of bodily and spiritual contamination and degradation. It supports the virtues of sanctity, nobility, and cleanliness.
- **p1:** The virtues of sanctity, nobility, and cleanliness rest on intuitions about avoiding bodily and spiritual contamination and degradation.

### Thin Morality (`thin_morality`)

- **Original:** A moral judgment or concern which is voiced without clearly referring to one of the six moral domains.
- **p0:** A moral judgment or concern that is expressed without clearly pointing to any one of the six moral domains.
- **p1:** Voicing a moral judgment or concern while not clearly referring to any of the six moral domains.

### Non-Moral (`non_moral`)

- **Original:** The comment lacks moral sentiment.
- **p0:** The comment contains no moral sentiment.
- **p1:** No moral sentiment is present in the comment.

## annomi_client

### Change talk (`change`)

- **Original:** The client's own language in favor of the change goal.
- **p0:** Language coming from the client that supports the change goal.
- **p1:** What the client says in favor of the change goal, in the client's own words.

### Sustain talk (`sustain`)

- **Original:** The client's language giving reasons against changing or for maintaining the status quo.
- **p0:** Client statements that offer reasons not to change or to keep things as they are.
- **p1:** Language from the client that argues against changing, or in support of the status quo.

### Neutral talk (`neutral`)

- **Original:** No indication of client inclination toward or away from change.
- **p0:** The client shows no leaning either toward change or away from it.
- **p1:** There is no sign of the client being inclined in the direction of change or against it.

## annomi_therapist

### Reflection (`reflection`)

- **Original:** A reflective listening statement made by the clinician in response to client statements. Reflections may introduce new meaning or material, but they essentially capture and return to clients something about what they have just said. Reflections may be either Simple or Complex.
- **p0:** A reflective listening statement by the clinician that responds to what the client has said. A reflection can bring in new meaning or material, yet it fundamentally gathers up something the client just said and gives it back to them. Reflections can be Simple or Complex.
- **p1:** Either Simple or Complex, a reflection is a reflective listening statement from the clinician answering the client's statements; although it may introduce new meaning or material, its essence is to capture something the client has just said and return it to them.

### Question (`question`)

- **Original:** A question from the clinician (open, closed, evocative, fact-finding, etc.).
- **p0:** The clinician asks the client a question, whether open, closed, evocative, fact-finding, or another kind.
- **p1:** Any question put by the clinician counts (open, closed, evocative, fact-finding, etc.).

### Therapist input (`therapist_input`)

- **Original:** The clinician provides input to the client: information, advice, negotiation, or options.
- **p0:** The clinician gives the client input, in the form of information, advice, negotiation, or options.
- **p1:** Input from the clinician to the client: information, advice, negotiation, or options.

### Other (`none`)

- **Original:** The utterance contains none of reflection, question, or therapist input (for example, a brief acknowledgement, greeting, or closing).
- **p0:** The utterance includes no reflection, no question, and no therapist input (for instance, a brief acknowledgement, a greeting, or a closing).
- **p1:** Other applies when an utterance is none of the following: reflection, question, or therapist input; examples are a brief acknowledgement, greeting, or closing.

## talkmoves_teacher

### Keeping Everyone Together (`keeping_together`)

- **Original:** Teacher prompts for students to be active listeners are coded as Keeping Everyone Together. This code is used for yes/no questions, managing turns, getting students' attention, keeping students on task, asking a student to read a question, call-and-response, and asking students to repeat each other.
- **p0:** When a teacher prompts students to listen actively, the sentence is coded as Keeping Everyone Together. The code covers yes/no questions, managing turns, capturing students' attention, keeping students on task, asking a student to read a question, call-and-response, and asking students to repeat one another.
- **p1:** Yes/no questions, turn management, attention-getting, keeping students on task, asking a student to read a question, call-and-response, and asking students to repeat each other all fall under Keeping Everyone Together, which is the code for teacher prompts that get students to listen actively.

### Getting Students to Relate (`relate`)

- **Original:** Teacher prompts for student contributions that relate to another student's work or contribution are coded as Getting Students to Relate. These prompts must go beyond asking a student to repeat another student's utterance. This code is used when teachers ask students to comment on or to agree/disagree with another's work or ideas.
- **p0:** A teacher prompt that asks for a student contribution tied to another student's work or contribution is coded as Getting Students to Relate. The prompt has to do more than ask a student to repeat what another student said. It applies when teachers ask students to comment on, or to agree or disagree with, someone else's work or ideas.
- **p1:** Getting Students to Relate is used when a teacher asks students to comment on another's work or ideas, or to agree/disagree with them. Merely asking a student to repeat another student's utterance does not suffice; the prompt must go further and elicit a contribution connected to another student's work or contribution.

### Restating (`restating`)

- **Original:** The teacher repeating all or part of what a student says word for word, without adding anything or changing any of the wording, is coded as Restating.
- **p0:** Restating is the code for when the teacher says all or part of a student's words exactly as spoken, with nothing added and no wording changed.
- **p1:** If a teacher repeats all or part of a student's utterance word for word, adding nothing and altering no wording, the code is Restating.

### Pressing for Accuracy (`press_accuracy`)

- **Original:** Teacher prompts for an answer related to a mathematical procedure or for a short math-related answer are coded as Pressing for Accuracy. The teacher may be prompting students to walk through problem-solving procedures, consider language accuracy, or define elements of a problem. This code also may be applied when the teacher asks students to write their answers at the board in front of the class.
- **p0:** Teacher prompts that ask for an answer tied to a mathematical procedure, or for a short math-related answer, are coded as Pressing for Accuracy. The teacher might be asking students to walk through problem-solving procedures, think about language accuracy, or define elements of a problem. The code may also be used when the teacher has students write their answers at the board in front of the class.
- **p1:** Pressing for Accuracy covers teacher prompts seeking an answer about a mathematical procedure or a short math-related answer: for instance, having students walk through problem-solving procedures, attend to language accuracy, or define elements of a problem. It may also be applied when students are asked to write their answers at the board in front of the class.

### Revoicing (`revoicing`)

- **Original:** The teacher repeating what a student says AND adding on to or changing the wording in any way is coded as Revoicing. This includes when the teacher is correcting a student and when a teacher repeats a student's utterance in another language.
- **p0:** Revoicing is coded when the teacher repeats what a student says and also adds to it or changes its wording in any way. This includes teacher corrections of a student and teacher repetition of a student's utterance in another language.
- **p1:** When a teacher repeats a student's utterance but also adds to it or alters the wording in any manner, the code is Revoicing; this covers correcting a student as well as repeating a student's utterance in another language.

### Pressing for Reasoning (`press_reasoning`)

- **Original:** Teacher prompts for students to provide evidence, explain their idea, or share their thinking behind a decision are coded as Press for Reasoning. The teacher may also be prompting for students to connect concepts or compare mathematical representations.
- **p0:** Teacher prompts asking students to give evidence, explain their idea, or share the thinking behind a decision are coded as Press for Reasoning. The teacher may also be prompting students to connect concepts or compare mathematical representations.
- **p1:** Press for Reasoning is the code for prompts in which the teacher asks students to explain an idea, offer evidence, or share the thinking underlying a decision, and may also include prompts for students to connect concepts or compare mathematical representations.

### None (`none`)

- **Original:** The sentence is not one of the teacher talk moves above (what is not coded).
- **p0:** A sentence that is not any of the teacher talk moves described above (i.e., what is not coded) receives the code None.
- **p1:** None is assigned to any sentence that does not match one of the other teacher talk move codes, serving as the code for what is not coded.

## talkmoves_student

### Relating to Another Student (`relating`)

- **Original:** When a student uses their classmate's idea as a basis for their own idea, these utterances are coded as Relating to Another Student. This code is applied when a student comments on or asks about another's work, a student answers another student's question, students agree/disagree with another's ideas, or a student repeats another student's utterance.
- **p0:** An utterance is coded as Relating to Another Student when the speaker takes a classmate's idea and builds their own idea on it. The code covers commenting on or asking about a peer's work, answering a peer's question, agreeing or disagreeing with a peer's ideas, and repeating a peer's utterance.
- **p1:** Comments on or questions about a classmate's work, answers to a classmate's question, agreement or disagreement with a classmate's ideas, and repetitions of a classmate's utterance are all coded as Relating to Another Student, because in each case the student is using a peer's idea as the foundation for their own.

### Asking for More Information (`asking_info`)

- **Original:** Student utterances that create an opportunity for more discussion about a math idea by saying they want more information are coded as Asking for More Information. This code is also used when students express confusion.
- **p0:** Utterances in which a student states that they want more information, thereby opening up further discussion of a math idea, are coded as Asking for More Information. Expressions of confusion by students also receive this code.
- **p1:** Code a student utterance as Asking for More Information if it expresses confusion, or if it says the student wants more information and so creates a chance for further discussion about a math idea.

### Making a Claim (`claim`)

- **Original:** When a student makes a mathematical claim or makes a factual statement about a math concept or idea, the utterance is coded as Making a Claim. This code is also used when a student shares how they got their answer in a simple statement, defines an element or provides an equation as an answer to a problem, disagrees with the teacher, or recognizes an error in their thinking. If the student does any of the above and includes evidence or reasoning, it is coded as Providing Evidence or Reasoning.
- **p0:** An utterance is coded as Making a Claim when a student asserts a mathematical claim or states a fact about a math concept or idea. The code also covers a student describing how they reached an answer in a simple statement, defining an element or giving an equation as the answer to a problem, disagreeing with the teacher, or acknowledging an error in their own thinking. If any of these includes evidence or reasoning, the code is Providing Evidence or Reasoning instead.
- **p1:** Whenever a student does any of the following with evidence or reasoning included, the code is Providing Evidence or Reasoning rather than Making a Claim. Without evidence or reasoning, these are coded as Making a Claim: stating a mathematical claim or a fact about a math concept or idea, explaining in a simple statement how they got their answer, defining an element or giving an equation as an answer to a problem, disagreeing with the teacher, or recognizing an error in their thinking.

### Providing Evidence or Reasoning (`evidence`)

- **Original:** Student utterances that suggest a deeper understanding of a math concept are coded as Providing Evidence or Reasoning. This code is used when a student describes a procedure with multiple steps, provides reasoning for their ideas, notices a pattern, answers a 'why' question, uses an if/then statement, or compares two or more mathematical representations.
- **p0:** Student utterances that indicate a deeper understanding of a math concept are coded as Providing Evidence or Reasoning. This includes describing a multi-step procedure, giving reasons for ideas, noticing a pattern, answering a 'why' question, using an if/then statement, or comparing two or more mathematical representations.
- **p1:** Code an utterance as Providing Evidence or Reasoning when it shows a deeper understanding of a math concept, specifically when the student gives reasoning for their ideas, answers a 'why' question, notices a pattern, uses an if/then statement, compares two or more mathematical representations, or describes a procedure that has multiple steps.

### None (`none`)

- **Original:** The sentence is not one of the student talk moves above.
- **p0:** The sentence does not match any of the student talk moves listed above.
- **p1:** A sentence that falls under none of the preceding student talk moves is coded as None.

## dunivin

### Scholar (`scholar`)

- **Original:** Applies when Du Bois is described as a scholar or intellectual, especially in connection to Black politics, racial identity, or social theory. When Du Bois is invoked through his ideas on social theory, he should be classified as a scholar, not an activist, unless it is call to action, related to his organizing, or other non-scholarly political activity. Do not apply when Du Bois is merely the focus the context of historical and academic study or his scholarship is only implied by loose connections to other scholars.
- **p0:** Use this code when Du Bois is portrayed as a scholar or intellectual, particularly with regard to Black politics, racial identity, or social theory. If Du Bois is brought up through his social-theory ideas, code him as a scholar rather than an activist, except where the passage involves a call to action, his organizing, or some other political activity that is not scholarly. Do not use it when Du Bois is merely the subject within historical and academic study, or when his scholarship is only suggested through loose links to other scholars.
- **p1:** Do not apply this code if Du Bois is just the topic of historical and academic study, or if his scholarship is only hinted at through loose associations with other scholars. Apply it when Du Bois is described as a scholar or intellectual, especially concerning Black politics, racial identity, or social theory. Where his social-theory ideas are what is invoked, classify him as a scholar and not an activist, unless the passage contains a call to action, refers to his organizing, or describes other non-scholarly political activity.

### Activist (`activist`)

- **Original:** Apply this code when Du Bois is explicitly called an "activist" or "leader", or when his political or social activism is either explicitly noted or clearly implied through context. Examples include being mentioned in the context of leadership, activism, developing activist organizations, giving public speeches, participating in meetings with politicians and organizers, running for office, or promoting a candidate, organization, or initiative.
- **p0:** Use this code if Du Bois is explicitly labeled an "activist" or "leader", or if his political or social activism is stated outright or clearly implied by context. Examples: references to leadership, activism, building activist organizations, delivering public speeches, meeting with politicians and organizers, running for office, or backing a candidate, organization, or initiative.
- **p1:** Context in which Du Bois is mentioned as leading, doing activism, developing activist organizations, giving public speeches, meeting with politicians and organizers, running for office, or promoting a candidate, organization, or initiative qualifies for this code. It also applies whenever he is explicitly named an "activist" or "leader", or when his political or social activism is explicitly noted or clearly implied by the context.

### Monumental Memorialization (`monumental_memorialization`)

- **Original:** When an enduring cultural object is named after Du Bois. Such objects include  prizes/awards, named professorships, buildings or rooms, geographical features, institutes, schools, or activist organizations. Do not apply when Du Bois is mentioned in the title of a book or theater production.
- **p0:** Use this code when a lasting cultural object bears Du Bois's name. Such objects include prizes or awards, named professorships, buildings or rooms, geographical features, institutes, schools, or activist organizations. Do not use it when Du Bois's name only appears in the title of a book or theater production.
- **p1:** A book or theater production with Du Bois in its title does not qualify for this code. It applies when an enduring cultural object is named for Du Bois, whether a prize/award, named professorship, building or room, geographical feature, institute, school, or activist organization.

### Mention of Scholarly Work (`scholarly_work`)

- **Original:** Apply this code when academic works or major theoretical concepts by W.E.B. Du Bois are mentioned or quoted. This includes explicit naming or direct quotations of his writings and references to his key academic ideas, even if unnamed, provided they are clearly attributed to him. Only use when a quote comes from a scholarly work; use context to determine whether a quote comes from a scholarly work, such as a history or social theory, or some other piece, such as a letter or speech. Avoid using this code for general references to Du Bois’s influence, body of writings, or activities outside of his scholarly work. Do not apply it when discussing others' work, unless Du Bois's scholarly concepts or writings are explicitly and centrally mentioned.
- **p0:** Use this code when W.E.B. Du Bois's academic works or major theoretical concepts are mentioned or quoted, including explicit naming or direct quotation of his writings and references to his key academic ideas, even when unnamed, so long as they are clearly attributed to him. A quotation counts only if it comes from a scholarly work; rely on context to judge whether it is from scholarship, such as history or social theory, versus another kind of text, such as a letter or speech. Do not use it for general references to Du Bois's influence, body of writings, or activities outside his scholarly work. Do not use it when others' work is discussed, unless Du Bois's scholarly concepts or writings are explicitly and centrally mentioned.
- **p1:** This code is not for general references to Du Bois's influence, his body of writings, or activities beyond his scholarly work, nor for discussion of others' work unless Du Bois's scholarly concepts or writings are explicitly and centrally mentioned. It is for passages that mention or quote academic works or major theoretical concepts by W.E.B. Du Bois: explicit naming or direct quotes of his writings, or references to his key academic ideas (even unnamed) clearly attributed to him. A quote qualifies only if it is from a scholarly work, which context should reveal (e.g., a history or social theory versus a letter or speech).

### Social/Political Advocacy (`advocacy`)

- **Original:** This code applies when a passage mentions or implies any form of social or political activism, advocacy, critique, or discourse, including discussions about current or historical social problems. This includes not only direct activism of Du Bois and others, but also the framing and challenging of social norms, historical narratives, and racial or cultural identities. Apply this code when Du Bois's work, persona, or ideas are invoked in discussions that critically engage with Black identity, positionality, or broader systemic circumstances of Black people. Adjacency to other activists such as inclusion in a list, is insufficient; advocacy must be explicitly mentioned in the passage.
- **p0:** Use this code when a passage mentions or implies any kind of social or political activism, advocacy, critique, or discourse, including talk of current or historical social problems. This covers direct activism by Du Bois and others as well as the framing and challenging of social norms, historical narratives, and racial or cultural identities. Use it as well when Du Bois's work, persona, or ideas are called upon in discussions that critically engage with Black identity, positionality, or the wider systemic circumstances of Black people. Merely being next to other activists, such as appearing in a list, is not enough; advocacy must be explicitly mentioned in the passage.
- **p1:** Advocacy must be explicitly mentioned in the passage; appearing alongside other activists, as in a list, does not suffice. Beyond that, this code fits any passage that mentions or implies social or political activism, advocacy, critique, or discourse, including about current or historical social problems, whether direct activism by Du Bois and others or the framing and challenging of social norms, historical narratives, and racial or cultural identities. It also fits invocations of Du Bois's work, persona, or ideas within discussions critically engaging with Black identity, positionality, or broader systemic circumstances of Black people.

### Coalition Building (`coalition_building`)

- **Original:** Du Bois is described as an agent establishing his reputation through organizational and institutional sponsorship.
- **p0:** Du Bois is depicted as an agent who establishes his reputation by means of organizational and institutional sponsorship.
- **p1:** Use this code when Du Bois is portrayed as an agent whose reputation is established through sponsorship from organizations and institutions.

### Out of the Mouth of Academics (`mouth_of_academics`)

- **Original:** Apply this code when a specific member of an academic organization is engaging with Du Bois’s work or Du Bois as representing a concept. Apply also when an activist organization has named itself or a subdivision of itself after Du Bois, such as an institute or named professorship. It is not sufficient to use this code when Du Bois is described as a member of an academic organization. This code represents when a member of an academic organization discusses Du Bois. It is not sufficient to pressume that the entity discussing or promoting Du Bois is academic.
- **p0:** Use this code when a specific member of an academic organization engages with Du Bois's work, or with Du Bois as representing a concept. Use it too when an activist organization has named itself, or a subdivision of itself such as an institute or named professorship, after Du Bois. Describing Du Bois as a member of an academic organization is not enough. The code captures a member of an academic organization discussing Du Bois; it is not enough to presume that the entity discussing or promoting Du Bois is academic.
- **p1:** This code captures a member of an academic organization discussing Du Bois, either engaging with his work or with him as representing a concept. It also applies where an activist organization has named itself or a subdivision (such as an institute or named professorship) after Du Bois. It does not apply merely because Du Bois is described as belonging to an academic organization, nor may you presume that whoever discusses or promotes Du Bois is academic.

### Out of the Mouth of Activists (`mouth_of_activists`)

- **Original:** Apply this code when an individual described as a leader, activist, or politician, or as member of a specific political or activist organization organization (e.g., political parties, the NAACP, the Black church, Black Lives Matter) references or draws upon W.E.B. Du Bois’s work or legacy. This includes instances where the person's role as a leader, activist, or politician is implied through their actions or affiliations, even if not explicitly stated. The code also applies when governments, political parties, or activist organizations connect their agenda to Du Bois, such as commemorating Du Bois by naming initiatives (like foundations or prizes) after him. Do not apply Du Bois is mentioned as a member of an organization, unless a representative organization is referencing Du Bois' membership to connect their agenda to Du Bois.
- **p0:** Use this code when someone described as a leader, activist, or politician, or as a member of a specific political or activist organization (e.g., political parties, the NAACP, the Black church, Black Lives Matter), references or draws on W.E.B. Du Bois's work or legacy. This includes cases where the person's role as leader, activist, or politician is implied by their actions or affiliations, even if not stated. It also applies when governments, political parties, or activist organizations tie their agenda to Du Bois, such as by commemorating him through initiatives (like foundations or prizes) named after him. Do not use it when Du Bois is mentioned as a member of an organization, unless a representative of that organization cites his membership to link their agenda to him.
- **p1:** Where Du Bois is simply mentioned as an organization's member, do not apply this code, unless an organization's representative cites that membership to connect their agenda to him. Otherwise, apply it when a leader, activist, politician, or member of a specific political or activist organization (e.g., political parties, the NAACP, the Black church, Black Lives Matter) references or draws on W.E.B. Du Bois's work or legacy, including where their role as leader, activist, or politician is implied through actions or affiliations though not explicitly stated. It further covers governments, political parties, or activist organizations linking their agenda to Du Bois, for instance by commemorating him via initiatives (like foundations or prizes) named after him.

### Collective Synecdoche (`collective_synecdoche`)

- **Original:** Mentioned with other famous intellectuals, activists, or public figures in order to represent some facet of a culture, era, or ideology. Examples include representing race scholarship, civil rights activism, left political leaders, Black excellence, and 20th century political commentators.
- **p0:** Du Bois is named together with other famous intellectuals, activists, or public figures so as to stand for some aspect of a culture, era, or ideology. Examples: standing for race scholarship, civil rights activism, left political leaders, Black excellence, and 20th century political commentators.
- **p1:** Use this code when Du Bois is grouped with other famous intellectuals, activists, or public figures in order to represent a facet of a culture, era, or ideology, such as race scholarship, civil rights activism, left political leaders, Black excellence, or 20th century political commentators.
