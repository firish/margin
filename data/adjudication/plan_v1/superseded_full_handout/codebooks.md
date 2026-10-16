# Codebooks

## mfrc (Reddit comment; any number of codes)

Which moral sentiments does this Reddit comment express? Several may apply.

_Source: Definitions verbatim from Trager et al. (2022), "The Moral Foundations Reddit Corpus", arXiv:2208.05545 (foundation descriptions given to annotators). Non-Moral is the paper's description of posts lacking moral sentiment._

### Care

Intuitions about avoiding emotional and physical damage or harm to others. It underlies virtues of kindness, gentleness, and nurturing, and vices of meanness, violence, and abuse.

### Equality

Intuitions about egalitarian treatment and equal outcome for all individuals and groups.

### Proportionality

Intuitions about individuals getting rewarded in proportion to their merit (i.e., effort, talent, or input).

### Loyalty

Intuitions about in-group cooperation and out-group competition. It underlies virtues of patriotism and self-sacrifice for the group.

### Authority

Intuitions about deference toward legitimate authorities and high-status individuals. It underlies virtues of leadership and respect for tradition.

### Purity

Intuitions about avoiding bodily and spiritual contamination and degradation. It underlies virtues of sanctity, nobility, and cleanliness.

### Thin Morality

A moral judgment or concern which is voiced without clearly referring to one of the six moral domains.

### Non-Moral

The comment lacks moral sentiment.

## annomi_client (client utterance; one code)

In this motivational interviewing session, what type of talk is the client's utterance with respect to the target behaviour change given in `setting`?

_Source: AnnoMI client talk types (Wu et al., ICASSP 2022) follow MISC. Change and sustain talk definitions use MITI 4.2.1 wording (Moyers et al.: "the client's own language in favor of the change goal"; "reasons against changing or for maintaining the status quo"). The MISC subcode descriptions and the AnnoMI-to-MISC mapping are from Sun et al. (LREC-COLING 2024, BiMISC), Tables 2-3. No examples are given for any code, so all three are described symmetrically._

### Change talk

The client's own language in favor of the change goal.

**Applies when:**
- Commitment: an agreement, intention, or obligation regarding future change.
- Taking steps: concrete steps the client has recently taken to make a change.
- Reason: rationale, basis, justification, or motive to make a change.
- Other: other statements that clearly reflect an intention to change.

### Sustain talk

The client's language giving reasons against changing or for maintaining the status quo.

**Applies when:**
- Commitment: an agreement, intention, or obligation not to change.
- Taking steps: concrete steps the client has recently taken away from change.
- Reason: rationale, basis, justification, or motive not to change.
- Other: other statements that clearly reflect an intention not to change.

### Neutral talk

No indication of client inclination toward or away from change.

**Applies when:**
- Follow/neutral: responses that show no inclination toward or away from change.
- Ask: asking for clarification or information.

## annomi_therapist (therapist utterance; one code)

In this motivational interviewing session, what is the main behaviour in the therapist's utterance? If it contains more than one behaviour, choose the main one.

_Source: AnnoMI therapist behaviours (Wu et al., ICASSP 2022; AnnoMI README for the input subtypes). Reflection, question, and information wording from MITI 4.2.1 (Moyers et al.) sections E.4.a, E.4.d, E.4.e. "Other" in AnnoMI is used exactly when none of reflection, question, or input is present (verified on AnnoMI-full: 2,143 of 2,143 'other' rows have all three flags false)._

### Reflection

A reflective listening statement made by the clinician in response to client statements. Reflections may introduce new meaning or material, but they essentially capture and return to clients something about what they have just said. Reflections may be either Simple or Complex.

**Applies when:**
- Simple reflections typically convey understanding or facilitate client-clinician exchanges. These reflections add little or no meaning (or emphasis) to what clients have said.
- Complex reflections typically add substantial meaning or emphasis to what the client has said, conveying a deeper or more complex picture of what the client has said.

**Examples:**
- You're furious about this.
- This is the last straw for you.
- You are pretty discouraged about this.
- You don't know why you're sabotaging yourself.

### Question

A question from the clinician (open, closed, evocative, fact-finding, etc.).

### Therapist input

The clinician provides input to the client: information, advice, negotiation, or options.

**Applies when:**
- Information: the clinician gives information, educates, provides feedback, or expresses a professional opinion without persuading, advising, or warning.
- Advice: the clinician advises the client on what to do.
- Negotiation: the clinician negotiates goals or a plan with the client.
- Options: the clinician offers the client options to choose from.

### Other

The utterance contains none of reflection, question, or therapist input (for example, a brief acknowledgement, greeting, or closing).

## talkmoves_teacher (teacher sentence; one code)

Which teacher talk move is this sentence from a K-12 mathematics lesson? Teacher talk moves are teacher utterances meant to elicit a student response (regardless of whether they actually do), or that repeat, elaborate on, or build off of student contributions.

_Source: TalkMoves Coding Manual (Suresh et al., SumnerLab/TalkMoves, CC BY-NC-SA 4.0), sections 1.3-1.6, verbatim. Sentences tagged "7 - context" in the data are excluded: the manual does not define it._

### Keeping Everyone Together

Teacher prompts for students to be active listeners are coded as Keeping Everyone Together. This code is used for yes/no questions, managing turns, getting students' attention, keeping students on task, asking a student to read a question, call-and-response, and asking students to repeat each other.

**Applies when:**
- Yes/no questions (that are not related to another student's idea/contribution)
- Teacher utterances that manage student turns or interactions in a general way
- Teacher utterances that are intended to keep students' attention
- Questions intended to keep a student on task
- Asking a student to read a problem/instructions aloud or to repeat a teacher utterance
- The teacher prompting a call-and-response utterance
- Asking the students if they heard or can repeat what another student or teacher said

**Examples:**
- So x equals five dollars, right?
- Go ahead.
- Can you repeat what you just said?
- Turn and talk to your partner.
- Are you finished?
- Can you read the problem out loud?
- What did she say?

### Getting Students to Relate

Teacher prompts for student contributions that relate to another student's work or contribution are coded as Getting Students to Relate. These prompts must go beyond asking a student to repeat another student's utterance. This code is used when teachers ask students to comment on or to agree/disagree with another's work or ideas.

**Applies when:**
- Asking students to comment or elaborate on other students' contributions
- Prompting students to take a position/stance or talk about their own thinking in relation to that of another student in the class

**Examples:**
- How do you feel about what they said?
- Would you like to add on to what he said?
- Does anyone understand how she solved the problem?
- Do you agree or disagree with him?
- Who is right?
- Did anyone else have three?

### Restating

The teacher repeating all or part of what a student says word for word, without adding anything or changing any of the wording, is coded as Restating.

**Examples:**
- (after S: The same size and shape but it moves to a different position.) It moves to a different position.
- (after S: An exponent) Exponent.
- (after S: It's four million and then the two.) Four million two.

### Pressing for Accuracy

Teacher prompts for an answer related to a mathematical procedure or for a short math-related answer are coded as Pressing for Accuracy. The teacher may be prompting students to walk through problem-solving procedures, consider language accuracy, or define elements of a problem. This code also may be applied when the teacher asks students to write their answers at the board in front of the class.

**Applies when:**
- The teacher requesting an answer to a problem
- Prompting a student to talk through a problem-solving procedure or possible variations on a problem
- Prompting for an answer related to language precision
- The teacher asking a student to define elements in an equation

**Examples:**
- What is the answer to number 2?
- What is 6 times 6?
- Can you write your answer on the board?
- How did you solve it?
- What is this called?
- What does x stand for?

### Revoicing

The teacher repeating what a student says AND adding on to or changing the wording in any way is coded as Revoicing. This includes when the teacher is correcting a student and when a teacher repeats a student's utterance in another language.

**Examples:**
- (after S: It had two.) So instead of one flat edge, it had two.
- (after S: An exponent.) The exponent.
- (after S: Oh, Company B.) It's Company B because that's the one that charges you $2.00 per minute.
- (after S: I got 2X minus Y.) It's not 2X.

### Pressing for Reasoning

Teacher prompts for students to provide evidence, explain their idea, or share their thinking behind a decision are coded as Press for Reasoning. The teacher may also be prompting for students to connect concepts or compare mathematical representations.

**Applies when:**
- Teacher prompts students to explain their ideas
- Teacher asks a student to connect two or more ideas or explain how a concept is related to the problem
- Teacher prompts students to compare across mathematical representations

**Examples:**
- Can you explain why?
- How did you know?
- Why does this method work?
- How are these ideas connected?
- How would that equation look on a graph?

### None

The sentence is not one of the teacher talk moves above (what is not coded).

**Applies when:**
- Teacher utterances that neither ask for a student contribution nor repeat all or part of a student contribution.
- Teacher utterances that are not related to the lesson (such as talking to students before or after a lesson).
- Teacher utterances that are not understandable (i.e., an important part of the transcript is blank).
- Teacher questions like "Okay?" or "Right?" if that is the complete utterance.
- Evaluative statements like "That's interesting," or "Good."
- Teacher utterances that seem like a talk move but do not fit into any of our categories.
- Teacher utterances explaining a concept generally or stating goals or objectives.
- Teacher utterances asking students to write something on their own paper.
- Teacher utterances addressing another teacher.
- Teacher questions that have no bearing on the math, e.g. "What should we call this street?"

## talkmoves_student (student sentence; one code)

Which student talk move is this sentence from a K-12 mathematics lesson? Student talk moves are discourse actions such as making claims, using reasoning, reacting to other students' ideas, and asking questions.

_Source: TalkMoves Coding Manual (Suresh et al., SumnerLab/TalkMoves, CC BY-NC-SA 4.0), sections 2.3-2.5, verbatim. The manual has no explicit "not coded" section for students; None is defined as the absence of the four moves._

### Relating to Another Student

When a student uses their classmate's idea as a basis for their own idea, these utterances are coded as Relating to Another Student. This code is applied when a student comments on or asks about another's work, a student answers another student's question, students agree/disagree with another's ideas, or a student repeats another student's utterance.

**Applies when:**
- Student comments on, adds on to, or asks about another student's utterance or work
- Student asks for confirmation or help on their own work from a peer
- Student answers another student's question
- Student agrees with/affirms or disagrees with/challenges what another student says
- Student rephrases or repeats another student's utterance

**Examples:**
- He drew the y-intercept.
- Her graph is different from mine.
- I agree with his answer.
- I don't think that is the way it goes.
- Did I do it right?
- (after T: Do you agree with her?) Yes.

### Asking for More Information

Student utterances that create an opportunity for more discussion about a math idea by saying they want more information are coded as Asking for More Information. This code is also used when students express confusion.

**Examples:**
- Why did you multiply?
- Where did you get the 5 from?
- What does Z mean?
- I don't understand.
- I need help.

### Making a Claim

When a student makes a mathematical claim or makes a factual statement about a math concept or idea, the utterance is coded as Making a Claim. This code is also used when a student shares how they got their answer in a simple statement, defines an element or provides an equation as an answer to a problem, disagrees with the teacher, or recognizes an error in their thinking. If the student does any of the above and includes evidence or reasoning, it is coded as Providing Evidence or Reasoning.

**Applies when:**
- Student provides an answer to a problem or math question
- Student shares how they got their answer in a simple statement
- Student defines elements of an equation or provides an equation as an answer to a problem
- Student disagrees with a teacher, including elliptical responses
- Student recognizes an error in their thinking
- Student offers an answer in the form of a yes/no question

**Examples:**
- (after T: What is 6 plus 6?) 12
- It's the same as number 6.
- I multiplied 8 and 5.
- Y is the number of cars.
- Oh wait, my answer is wrong.
- Would I divide 10 by 2?

### Providing Evidence or Reasoning

Student utterances that suggest a deeper understanding of a math concept are coded as Providing Evidence or Reasoning. This code is used when a student describes a procedure with multiple steps, provides reasoning for their ideas, notices a pattern, answers a 'why' question, uses an if/then statement, or compares two or more mathematical representations.

**Examples:**
- I divided 6 by 2 and got 3.
- I don't think so because you can't subtract 2 from zero.
- It is going up by 5 each time.
- (after T: Why can't we add these?) Because the denominators are not the same.
- If you double it, the shape is going to get bigger.
- 2x plus 8 is going to be a positive line on the graph.

### None

The sentence is not one of the student talk moves above.
