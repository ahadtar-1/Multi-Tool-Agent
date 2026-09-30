import os
from langsmith import Client
import asyncio
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv, find_dotenv
import langchain
import json
import pandas
from typing_extensions import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.agent_with_multiple_tools_opt import build_agent, run_agent_safely

_ = load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GOOGLE_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langsmith_tracing_message = os.getenv("LANGSMITH_TRACING_V2")
langsmith_project = os.getenv("LANGSMITH_PROJECT")

gemini_agent = build_agent()

gemini_threepointone_flashlitellm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", 
    temperature=1.0,
    model_kwargs={"generation_config": {"thinking_config": {"thinking_budget": 0}}}
)

client = Client()

examples = [
    {
        "inputs": {
            "question": "What is new in Windows Server 2012?",
        },
        "outputs": {
            "response": "Windows Server 2012 brings our company\'s experience building and operating public clouds to the server platform for private clouds. The new licensing and packaging makes it easier to manage workloads in highly virtualized public and private cloud environments. Windows Server 2012 will move to a consistent licensing model and will have common features enabling the reduction of editions. These include\n· Two editions, Standard and Datacenter.\n· Single licenses that cover up to two physical processors.\n· Editions differentiated by virtualization rights only (two for Standard; unlimited for Datacenter).",
            "trajectory": ["knowledge_retriever"],
            "category": "knowledge_retriever"
        }
    },
    {
        "inputs": {
            "question": "What is the difference between Windows Server 2012 Standard edition and Windows Server 2012 Datacenter edition?",
        },
        "outputs": {
            "response": "Both Standard and Datacenter editions provide the same set of features; the only thing that differentiates the editions is the number of Virtual Machines (VMs). A Standard edition license will entitle you to run up to two VMs on up to two processors (subject to the VM use rights outlined in the Product Use Rights document). A Datacenter edition license will entitle you to run an unlimited number of VMs on up to two processors.",
            "trajectory": ["knowledge_retriever"],
            "category": "knowledge_retriever"
        }
    },
    {
        "inputs": {
            "question": "Can I use one Standard license to cover a 1-processor server?",
        },
        "outputs": {
            "response": "Yes. The Standard edition license will allow you to license up to two physical processors on a single server; however it does not require that the server has two physical processors.",
            "trajectory": ["knowledge_retriever"],
            "category": "knowledge_retriever"
        }
    },
    {
        "inputs": {
            "question": "Can I assign a Windows Server 2012 license to a virtual machine?",
        },
        "outputs": {
            "response": "No. A license is assigned to the physical server. Each license will cover up to two physical processors.",
            "trajectory": ["knowledge_retriever"],
            "category": "knowledge_retriever"
        }
    },
    {
        "inputs": {
            "question": "What are my licensing options for a Disaster Recovery server?",
        },
        "outputs": {
            "response": "If you are storing virtual machines for future use in a Disaster Recovery situation you will not need additional licensing for that server. Only when you run an instance on that server will a license be required (see the definition of running an instance below). You should be mindful that any of your replicated virtual instances, when running, need to be running on a server appropriately licensed to support that running instance.\nRunning Instance means an Instance of software that is loaded into memory and for which one or more instructions have been executed. (You \"Run an Instance\" of software by loading it into memory and executing one or more of its instructions.) Once running, an Instance is considered to be running (whether or not its instructions continue to execute) until it is removed from memory.\nThere are two ways in which you can license a server for Disaster Recovery, by purchasing a Windows Server license or by using the Cold Back-up for Disaster Recovery Software Assurance benefit. Cold Back-up for Disaster Recovery allows you to keep a backup server ready for use in case your primary (production) server fails due to earthquakes, floods or any kind of disaster. If a disaster strikes you can immediately switch over to the Cold Disaster Recovery server. In order to utilize this benefit you must comply with the follow terms:\n· The software in the Disaster Recovery server should comply with the use rights associated with the software.\n· The server cannot be in the same cluster as the production server.\n· The server cannot be turned on except for updates to the software (patching) or testing. The server may of course be turned on in the event of a disaster for Disaster Recovery.\n· The Disaster Recovery server and the production instances may run concurrently while recovering from a disaster. At all other times the Disaster Recovery server should be switched off except as above\nRemember that in order to utilize this Software Assurance benefit, all licenses in use must have active Software Assurance coverage. This includes any CALs required to access the Disaster Recovery server. This benefit ends when Software Assurance coverage on your licenses ends. You can find more information about the use rights for this benefit in the Product Use Rights document.",
            "trajectory": ["knowledge_retriever"],
            "category": "knowledge_retriever"
        }
    },
    {
        "inputs": {
            "question": "Can I attach another license of a different version or edition of Windows Server to increase my virtualization rights?",
        },
        "outputs": {
            "response": "Yes, you can assign additional Windows Server licenses to a server to increase your virtualization rights. However the newly assigned licenses will need to adhere to their associated licensing rules. For example, if you have a Windows Server Enterprise edition license on a four processor server and want to attach Windows Server 2012 Standard edition licenses to increase your virtualization rights, you will need to ensure that all processors on that server are licensed with Windows Server 2012 Standard edition license as well, which will require a total of two Windows Server 2012 Standard licenses (as each license covers up to two processors).",
            "trajectory": ["knowledge_retriever"],
            "category": "knowledge_retriever"
        }
    },
    {
        "inputs": {
            "question": "What are the different editions available with Windows Server 2012 Essentials?",
        },
        "outputs": {
            "response": "There is only one edition-Windows Server 2012 Essentials. It is a flexible offering that provides a platform for running on-premises or cloud-based workloads.",
            "trajectory": ["knowledge_retriever"],
            "category": "knowledge_retriever"
        }
    },
    {
        "inputs": {
            "question": "Will there be a next version of Windows Small Business 2011 Premium Add-on?",
        },
        "outputs": {
            "response": "No. Windows Small Business Server 2011 Premium Add-on, which includes SQL Server and Windows Server as component products, will be the final such Windows Server offering.",
            "trajectory": ["knowledge_retriever"],
            "category": "knowledge_retriever"
        }
    },
    {
        "inputs": {
            "question": "What are some of the features that are now available in Windows Server 2012 Essentials?",
        },
        "outputs": {
            "response": "Windows Server 2012 Essentials incorporates best-of-breed 64-bit product technologies to deliver a server environment well-suited for the vast majority of small businesses. The product technologies include:\n· Windows Server 2012 operating system\n· Data protection\n· \"Anywhere\" access\n· Health monitoring\n· Workload flexibility\n· Extensibility\n· Add-ons for many small business solutions, including a connector to Office 365\nCustomers can use Windows Server 2012 Essentials as a platform to run critical line-of-business applications and other on-premises workloads. It can also provide an integrated management experience when running cloud-based applications and services, such as email, collaboration, online backup, and more.",
            "trajectory": ["knowledge_retriever"],
            "category": "knowledge_retriever"
        }
    },
    {
        "inputs": {
            "question": "Can I move Windows Server 2012 licenses and images between Hyper-V and Azure?",
        },
        "outputs": {
            "response": "Windows Server 2012 licenses just like Windows Server 2008 R2 are not eligible for the license mobility benefits under Software Assurance. You can continue to take advantage of the license mobility rights for other server applications, however Windows Server will continue to be purchased separately from the service provider or Azure.",
            "trajectory": ["knowledge_retriever"],
            "category": "knowledge_retriever"
        }
    },
    {
        "inputs": {
            "question": "What is 1589 + 3421?",
        },
        "outputs": {
            "response": "The sum of 1589 and 3421 is 5010.",
            "trajectory": ["calculator"],
            "category": "calculator"
        }
    },
    {
        "inputs": {
            "question": "Calculate 458 multiplied by 12.",
        },
        "outputs": {
            "response": "The product of 458 multiplied by 12 is 5496.",
            "trajectory": ["calculator"],
            "category": "calculator"
        }
    },
    {
        "inputs": {
            "question": "What is 1024 divided by 8?",
        },
        "outputs": {
            "response": "One thousand twenty four divided by eight is one hundred twenty eight.",
            "trajectory": ["calculator"],
            "category": "calculator"
        }
    },
    {
        "inputs": {
            "question": "Evaluate 2 to the power of 10.",
        },
        "outputs": {
            "response": "The result of 2 to the power of 10 is 1024.",
            "trajectory": ["calculator"],
            "category": "calculator"
        }
    },
    {
        "inputs": {
            "question": "What is 500 minus 137?",
        },
        "outputs": {
            "response": "The result of 500 minus 137 is 363.",
            "trajectory": ["calculator"],
            "category": "calculator"
        }
    },
    {
        "inputs": {
            "question": "Calculate (45 + 15) * 3.",
        },
        "outputs": {
            "response": "The result of the calculation is 180.",
            "trajectory": ["calculator"],
            "category": "calculator"
        }
    },
    {
        "inputs": {
            "question": "What is 999 modulo 10?",
        },
        "outputs": {
            "response": "The result of 999 modulo 10 is 9.",
            "trajectory": ["calculator"],
            "category": "calculator"
        }
    },
    {
        "inputs": {
            "question": "Evaluate 12.5 times 4.2.",
        },
        "outputs": {
            "response": "The result of 12.5 times 4.2 is 52.5.",
            "trajectory": ["calculator"],
            "category": "calculator"
        }
    },
    {
        "inputs": {
            "question": "What is 7 raised to the power of 3?",
        },
        "outputs": {
            "response": "Seven raised to the power of three is three hundred forty-three.",
            "trajectory": ["calculator"],
            "category": "calculator"
        }
    },
    {
        "inputs": {
            "question": "Calculate 5000 / (25 * 2).",
        },
        "outputs": {
            "response": "The result of the calculation is 100.",
            "trajectory": ["calculator"],
            "category": "calculator"
        }
    },
    {
        "inputs": {
            "question": "What are the latest tech headlines today?",
        },
        "outputs": {
            "response": "Here are the latest technology headlines:\n\nAdvanced Micro Devices is acquiring Fei-Fei Li's artificial intelligence firm World Labs in a transaction valued at 8.2 billion dollars.\nProminent technology executives, including Mark Zuckerberg and Dario Amodei, are scheduled to meet with government officials.\nNew York City has subpoenaed Elon Musk and SpaceX AI in connection with an ongoing artificial intelligence safety investigation.\nOpenAI has reportedly abandoned plans to release an upcoming model due to escalating safety concerns.\nMeta has launched Muse for Small Business as part of an expansion into enterprise artificial intelligence solutions.",
            "trajectory": ["web_search"],
            "category": "web_search"
        }
    },
    {
        "inputs": {
            "question": "Who won the most recent Super Bowl?",
        },
        "outputs": {
            "response": "The Seattle Seahawks won the most recent Super Bowl, defeating the New England Patriots in Super Bowl LX",
            "trajectory": ["web_search"],
            "category": "web_search"
        }
    },
    {
        "inputs": {
            "question": "What is the current stock price of Microsoft?",
        },
        "outputs": {
            "response": "The current stock price of Microsoft is 508.84 dollars per share.",
            "trajectory": ["web_search"],
            "category": "web_search"
        }
    },
    {
        "inputs": {
            "question": "What major space missions launched in 2026?",
        },
        "outputs": {
            "response": "Several major space missions launched in 2026, highlighted by significant crewed flights and scientific undertakings:\n\n1. Artemis II: Launched on April 1, 2026, this landmark NASA mission carried four astronauts on a crewed lunar flyby test using the Space Launch System and Orion spacecraft. It marked the first crewed journey beyond low Earth orbit since the Apollo program in 1972.\n2. Pandora Space Telescope: Launched by NASA on January 11, 2026, this small space telescope mission was deployed to study exoplanet atmospheres, accompanied by auxiliary CubeSat telescopes like SPARCS and BlackCAT.",
            "trajectory": ["web_search"],
            "category": "web_search"
        }
    },
    {
        "inputs": {
            "question": "Who is the current Prime Minister of the United Kingdom?",
        },
        "outputs": {
            "response": "The current Prime Minister of the United Kingdom is Andy Burnham.",
            "trajectory": ["web_search"],
            "category": "web_search"
        }
    },
    {
        "inputs": {
            "question": "What were the major weather updates in Lahore this week?",
        },
        "outputs": {
            "response": "The weather in Lahore this week has been characterized by warm and clear conditions with high temperatures. Daytime temperatures reached up to forty degrees Celsius earlier in the week before experiencing a slight decrease, with temperatures ranging around thirty-five to thirty-seven degrees Celsius and overnight lows dropping to about twenty-two to twenty-five degrees Celsius. Skies remained mostly sunny and clear with stable atmospheric conditions.",
            "trajectory": ["web_search"],
            "category": "web_search"
        }
    },
    {
        "inputs": {
            "question": "What movies are currently leading the box office?",
        },
        "outputs": {
            "response": "The leading movies at the box office in the United States currently include Resident Evil, Practical Magic 2, SpiderMan Brand New Day, The Odyssey, and Coyote vs Acme.",
            "trajectory": ["web_search"],
            "category": "web_search"
        }
    },
    {
        "inputs": {
            "question": "Who won the recent Nobel Prize in Physics?",
        },
        "outputs": {
            "response": "The Nobel Prize in Physics was awarded to John Clarke, Michel H. Devoret, and John M. Martinis for their discovery of macroscopic quantum mechanical tunnelling and energy quantisation in an electric circuit.",
            "trajectory": ["web_search"],
            "category": "web_search"
        }
    },
    {
        "inputs": {
            "question": "What are the latest developments in artificial intelligence released this month?",
        },
        "outputs": {
            "response": "The latest developments in artificial intelligence for this month include advancements in agentic capabilities, artificial intelligence safety, enterprise privacy, and self-improving systems. Additionally, major industry integrations have emerged, such as partnerships allowing select advertisers to run campaigns within conversational artificial intelligence platforms through major advertising networks",
            "trajectory": ["web_search"],
            "category": "web_search"
        }
    },
    {
        "inputs": {
            "question": "What is the current status of the war between USA and Iran?",
        },
        "outputs": {
            "response": "The conflict between the United States and Iran is currently characterized as a war of attrition. Following major military engagements earlier in the year, a memorandum of understanding and temporary ceasefires collapsed amid renewed fighting and the expiration of agreements. Both nations are currently trading limited military strikes alongside intensified economic sanctions by the United States, while regional governments continue efforts to foster diplomatic engagement and negotiate a durable settlement. Core disputes regarding regional security, economic sanctions, and Iran nuclear program remain unresolved.",
            "trajectory": ["web_search"],
            "category": "web_search"
        }
    },
    {
        "inputs": {
            "question": "Find the current stock price of Apple and multiply it by 10.",
        },
        "outputs": {
            "response": "The latest closing stock price for Apple is 329.40 dollars. Multiplying this value by 10 yields 3294.00 dollars.",
            "trajectory": ["web_search", "calculator"],
            "category": "web_search + calculator"
        }
    },
    {
        "inputs": {
            "question": "What is today's temperature in New York in Fahrenheit, and convert it to Celsius using a math expression?",
        },
        "outputs": {
            "response": "The forecasted daytime temperature in New York is seventy two degrees Fahrenheit. \n\nTo convert this temperature to Celsius, the following mathematical expression is used:\n(72 - 32) * 5 / 9\n\nThis calculation yields approximately twenty two point two two degrees Celsius.",
            "trajectory": ["web_search", "calculator"],
            "category": "web_search + calculator"
        }
    },
    {
        "inputs": {
            "question": "Look up the population of France and add 1,000 to it.",
        },
        "outputs": {
            "response": "The current population of France is estimated at 66746401. Adding 1000 to this figure results in 66747401.",
            "trajectory": ["web_search", "calculator"],
            "category": "web_search + calculator"
        }
    },
    {
        "inputs": {
            "question": "Find the latest conversion rate for USD to EUR, and calculate how much 500 USD is worth.",
        },
        "outputs": {
            "response": "The current conversion rate for one United States Dollar is approximately zero point eight six three five Euros. Based on this rate, five hundred United States Dollars is worth approximately four hundred thirty one point seven four Euros.",
            "trajectory": ["web_search", "calculator"],
            "category": "web_search + calculator"
        }
    },
    {
        "inputs": {
            "question": "Find the distance in miles between New York and London, and multiply it by 1.60934 to get kilometers.",
        },
        "outputs": {
            "response": "The distance in miles between New York and London is approximately 3461.34 miles. When this distance is multiplied by 1.60934, the result is approximately 5570.47 kilometers.",
            "trajectory": ["web_search", "calculator"],
            "category": "web_search + calculator"
        }
    },
    {
        "inputs": {
            "question": "Look up the current price of gold per ounce, and calculate the cost of 5 ounces.",
        },
        "outputs": {
            "response": "The current price of gold is approximately 4144.70 dollars per ounce. Multiplying this rate by 5 ounces results in a total cost of 20723.50 dollars",
            "trajectory": ["web_search", "calculator"],
            "category": "web_search + calculator"
        }
    },
    {
        "inputs": {
            "question": "Find the current year's global smartphone shipment total, and divide it by 12 to find the monthly average.",
        },
        "outputs": {
            "response": "According to data from the International Data Corporation, the global smartphone shipments for the current year of 2026 are forecast to reach approximately 1.09 billion units. \n\nDividing this total annual shipment of 1,090,000,000 units by 12 results in a monthly average of approximately 90,833,333 units.",
            "trajectory": ["web_search", "calculator"],
            "category": "web_search + calculator"
        }
    },
    {
        "inputs": {
            "question": "Find the current gas price per gallon in California, and calculate the cost for 15 gallons.",
        },
        "outputs": {
            "response": "The current average price for regular gasoline in California is approximately six dollars and thirty seven cents per gallon. Based on this rate, the total cost for fifteen gallons of gasoline is ninety five dollars and fifty five cents.",
            "trajectory": ["web_search", "calculator"],
            "category": "web_search + calculator"
        }
    },
    {
        "inputs": {
            "question": "What is the estimated retail price of a Windows Server 2012 Datacenter edition license, and what would be the total cost if I purchase 3 licenses?",
        },
        "outputs": {
            "response": "For your specific pricing, contact your Microsoft reseller. Actual prices may vary. Microsoft does not determine pricing or payment terms for licenses acquired through resellers.\n\nFor organizations considering volume licensing options, the Datacenter edition is priced at 4809 dollars under the Open No Level estimated retail price. In comparison, the Standard edition is significantly more affordable, with an estimated retail price of 882 dollars. This pricing information helps businesses determine which edition best fits their budget and needs when planning for software deployment.\n\nThe total cost for purchasing 3 licenses of the Windows Server 2012 Datacenter edition at the estimated retail price of 4809 dollars per license is 14427 dollars.",
            "trajectory": ["knowledge_retriever", "calculator"],
            "category": "knowledge_retriever + calculator"
        }
    },
    {
        "inputs": {
            "question": "How many physical processors are covered by a single Windows Server 2012 license, and if I have a server with 12 physical processors, how many licenses do I need?",
        },
        "outputs": {
            "response": "No. A license is assigned to the physical server. Each license will cover up to two physical processors. \n\nSince you have a server with twelve physical processors, and each license covers up to two physical processors, you will need six licenses to fully cover your server.",
            "trajectory": ["knowledge_retriever", "calculator"],
            "category": "knowledge_retriever + calculator"
        }
    },
    {
        "inputs": {
            "question": "How many Virtual Machines (VMs) does a single Windows Server 2012 Standard edition license entitle me to run, and if I assign 4 Standard licenses to a single server, what is the total number of VMs I am allowed to run?",
        },
        "outputs": {
            "response": "A single Windows Server 2012 Standard edition license entitles you to run two virtual machines. \n\nWith the Windows Server 2012 Standard edition licensing model, you can grow your virtualization environment by either stepping up your license to Datacenter edition if you have Software Assurance, or by simply buying additional Standard edition licenses and assigning them to the same physical server. For example, if you have a 2-processor server and want to run a total of 4 VMs, you can purchase two Standard edition licenses and assign them to the same server. Additional examples are shown in the table below.\n\nWhen using Standard edition licenses on a single server, each license allows for the creation of two virtual machines. For example, with one license, it is possible to run two virtual machines. If two licenses are applied, the total number of virtual machines increases to four. With three licenses, six virtual machines can be operated, and with four licenses, the server can support up to eight virtual machines. This pattern shows that each additional license consistently adds support for two more virtual machines on the server.",
            "trajectory": ["knowledge_retriever", "calculator"],
            "category": "knowledge_retriever + calculator"
        }
    },
    {
        "inputs": {
            "question": "What is the volume licensing estimated retail price for a Windows Server 2012 Standard edition license, and what is the combined total cost if I buy 5 Standard licenses?",
        },
        "outputs": {
            "response": "For your specific pricing, contact your Microsoft reseller. Actual prices may vary. Microsoft does not determine pricing or payment terms for licenses acquired through resellers.\n\nFor organizations considering volume licensing options, the Datacenter edition is priced at 4809 dollars under the Open No Level estimated retail price. In comparison, the Standard edition is significantly more affordable, with an estimated retail price of 882 dollars. This pricing information helps businesses determine which edition best fits their budget and needs when planning for software deployment.\n\nThe combined total cost for purchasing 5 Standard edition licenses is 4410 dollars.",
            "trajectory": ["knowledge_retriever", "calculator"],
            "category": "knowledge_retriever + calculator"
        }
    },
    {
        "inputs": {
            "question": "What is the estimated retail price of Windows Server 2012 Essentials, and how much more expensive is a Standard edition license compared to an Essentials license?",
        },
        "outputs": {
            "response": "For your specific pricing, contact your Microsoft reseller. Actual prices may vary. Microsoft does not determine pricing or payment terms for licenses acquired through resellers.\n\nEssentials edition is available for purchase through volume licensing at an estimated retail price of 501 dollars. In contrast, the Foundation edition is not available through volume licensing and can only be obtained as an Original Equipment Manufacturer or OEM version. This means that Foundation is typically pre-installed on hardware by manufacturers and is not sold separately through standard retail or volume licensing channels.\n\nFor your specific pricing, contact your Microsoft reseller. Actual prices may vary. Microsoft does not determine pricing or payment terms for licenses acquired through resellers.\n\nFor organizations considering volume licensing options, the Datacenter edition is priced at 4809 dollars under the Open No Level estimated retail price. In comparison, the Standard edition is significantly more affordable, with an estimated retail price of 882 dollars. This pricing information helps businesses determine which edition best fits their budget and needs when planning for software deployment.\n\nBased on the estimated retail prices, a Standard edition license is 381 dollars more expensive than an Essentials edition license.",
            "trajectory": ["knowledge_retriever", "calculator"],
            "category": "knowledge_retriever + calculator"
        }
    },
    {
        "inputs": {
            "question": "What are the top-trending electric vehicles on the market, and what are the upgrade options if I need more virtualization capacity on a Windows Server 2012 Standard license?",
        },
        "outputs": {
            "response": "With the Windows Server 2012 Standard edition licensing model, you can grow your virtualization environment by either stepping up your license to Datacenter edition if you have Software Assurance, or by simply buying additional Standard edition licenses and assigning them to the same physical server. For example, if you have a 2-processor server and want to run a total of 4 VMs, you can purchase two Standard edition licenses and assign them to the same server. Additional examples are shown in the table below.\n\nWhen using Standard edition licenses on a single server, each license allows for the creation of two virtual machines. For example, with one license, it is possible to run two virtual machines. If two licenses are applied, the total number of virtual machines increases to four. With three licenses, six virtual machines can be operated, and with four licenses, the server can support up to eight virtual machines. This pattern shows that each additional license consistently adds support for two more virtual machines on the server.",
            "trajectory": ["web_search", "knowledge_retriever"],
            "category": "web_search + knowledge_retriever"
        }
    },
    {
        "inputs": {
            "question": "What are the current global renewable energy generation statistics, and why did Microsoft transition Windows Server 2012 licensing to a processor model?",
        },
        "outputs": {
            "response": "Global renewable energy generation statistics indicate that renewables accounted for approximately 31.7 percent of total electricity generation worldwide, totaling 9,836 terawatt hours. Furthermore, total installed renewable capacity reached 5.2 terawatts, representing about 49.5 percent of the global total capacity.\n\nRegarding the transition of Windows Server 2012 licensing, Microsoft updated the model to align with System Center 2012 and the Core Infrastructure Server SKU. By making this change, Windows Server 2012, System Center 2012 and the Core Infrastructure Server (CIS) will all have consistent licensing model creating alignment across Microsoft infrastructure products. Having a single-licensing model will make it easier for you to buy the right product for your needs and to compare the cost of alternatives (such as individual products, the CIS SKU outside of ECI, ECI and SO on). Additionally, the new licensing model provides a single, familiar, and easy-to-track metric for all infrastructure products further reducing management overhead.",
            "trajectory": ["web_search", "knowledge_retriever"],
            "category": "web_search + knowledge_retriever"
        }
    },
    {
        "inputs": {
            "question": "Who won the most recent Formula 1 World Championship, and what are the Software Assurance migration rights for a Windows Server 2012 Enterprise agreement?",
        },
        "outputs": {
            "response": "Lando Norris won the most recent Formula One World Drivers Championship in the year 2025.\n\nEach Microsoft purchase program has different rules for your Software Assurance migration entitlement at the end of your enrollment. See the chart below.\n\nFor the Enterprise Agreement and Open Value programs, perpetual rights are granted to the current Windows Server 2012 edition at the time of release. In contrast, the Enterprise Agreement Subscription, Enrollment for Education Solutions – School Enrollment, Open Value Subscription, and Open Value Subscription – Education Solutions programs allow the use of the Windows Server 2012 edition during the enrollment period. At the end of enrollment, participants in these subscription-based programs have the option to buy out the original Windows Server 2008 R2 product at the original CPS price, or the new Windows Server 2012 product at the buy-out price listed at the time of enrollment expiration. Alternatively, they can renew their enrollment at the new Windows Server 2012 annual subscription price. This structure provides flexibility for organizations to either continue with updated software or secure perpetual rights through a buy-out option.\n\nUnder the Select or Open program, customers are granted perpetual rights to the current edition of Windows Server 2012 if they have Software Assurance at the time of release. This means that once they acquire the license, they can continue to use that specific version of Windows Server 2012 indefinitely, without the need for ongoing subscription payments or renewals.\nNote: The Enrollment for Core Infrastructure (ECI) follows the same rules as stated for the Enterprise Agreement in the chart above.",
            "trajectory": ["web_search", "knowledge_retriever"],
            "category": "web_search + knowledge_retriever"
        }
    },
    {
        "inputs": {
            "question": "What are the latest breakthroughs in quantum computing research and can I split my Windows Server 2012 license across multiple servers?",
        },
        "outputs": {
            "response": "Recent breakthroughs in quantum computing research include significant progress in fault tolerance, error suppression, and logical qubit scaling. Prominent achievements feature Google showcasing exponential error suppression with its Willow processor, Atom Computing demonstrating neutral atom logical qubit entanglement, and advancements in topological and Majorana qubits designed to resist errors at the hardware level.\n\nRegarding your question about Windows Server 2012, No. Each license can only be assigned to a single physical server.",
            "trajectory": ["web_search", "knowledge_retriever"],
            "category": "web_search + knowledge_retriever"
        }
    },
    {
        "inputs": {
            "question": "What major cybersecurity conferences are happening this year and is Enterprise edition going away as part of Windows Server 2012 and why?",
        },
        "outputs": {
            "response": "Major cybersecurity conferences taking place this year include the RSA Conference held in San Francisco, the Gartner Security and Risk Management Summit, and the ISC2 Security Congress in Aurora, Colorado. \n\nRegarding your question about Windows Server 2012, Yes. Enterprise edition will be retired as part of the Windows Server 2012 release. Windows Server 2012 Standard edition will include all the premium features previously included in Enterprise edition in the past and the price to purchase the rights to 4 instances of Windows Server 2012 will actually be less expensive than the price of Windows Server 2008 R2 Enterprise edition today. Due to these changes, we have been able to simplify the product lineup while reducing the price per instance of Windows Server for these customers.",
            "trajectory": ["web_search", "knowledge_retriever"],
            "category": "web_search + knowledge_retriever"
        }
    }
]

dataset_name = "Multi-Tool-Agent-Dataset"

if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        dataset_id=dataset.id,
        examples=examples
    )

# LLM-as-judge instructions
grader_instructions = """

</role>
You are an expert AI Agent Evaluator grading a test run.
</role>

<task>
You will be given a QUESTION, the GROUND TRUTH (correct) RESPONSE, the STUDENT RESPONSE, and the TEST CATEGORY.

<instructions>
Apply the following grading rules strictly based on the category:

1. KNOWLEDGE RETRIEVER (Single-Tool):
   - The student response MUST be an EXACT, character-for-character, verbatim match with the ground truth response. 
   - No summarization, rephrasing, or word alterations are allowed. 

2. CALCULATOR or WEB SEARCH (Single-Tool):
   - The student response does NOT need to be an exact match. Grade as True if the core meaning or numerical result is identical to the ground truth.

3. MULTI-TOOL QUERIES (Involving Knowledge Retriever):
   - The Ground Truth Response contains the raw text retrieved from the knowledge base which contains text related to Windows Server only. 
   - You must verify that the exact text block belonging to the knowledge retriever is present word-for-word, character-for-character inside the Student Response.
   - The remaining part of the Student Response (generated by the calculator or web search) does not need to be an exact match, but must be semantically correct.
   - If the knowledge retriever text block is modified, summarized, or rephrased within the student response, grade it as False.

Correctness:
True means the response meets all criteria.
False means it fails.
</instructions>

Explain your reasoning in a step-by-step manner before providing your conclusion.
</task>
"""

# LLM-as-judge output schema
class Grade(TypedDict):
    """Compare the expected and actual answers and grade the actual answer."""
    reasoning: Annotated[str, ..., "Explain your reasoning for whether the actual response is correct or not."]
    is_correct: Annotated[bool, ..., "True if the student response is mostly or exactly correct, otherwise False."]

# Judge LLM
grader_llm = gemini_threepointone_flashlitellm.with_structured_output(Grade, method="json_schema", strict=True)

# Evaluator function
async def final_answer_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """
    Evaluate if the final response meets category-specific matching rules, accounting for multi-tool verbatim constraints.
    """

    category = reference_outputs.get('category', 'unknown')
    trajectory = reference_outputs.get('trajectory', ['unknown'])

    user = f"""TEST CATEGORY: {category}
    TOOL TRAJECTORY: {trajectory}
    
    QUESTION: {inputs['question']}
    
    GROUND TRUTH RESPONSE: 
    {reference_outputs['response']}
    
    STUDENT RESPONSE TO EVALUATE: 
    {outputs['response']}
    
    EVALUATION NOTE FOR JUDGE: If the category involves 'knowledge_retriever', check the Student Response specifically for the exact, unedited documentation text found in the Ground Truth Response above.
    """

    grade = await grader_llm.ainvoke([
        {"role": "system", "content": grader_instructions}, 
        {"role": "user", "content": user}
    ])
    
    return grade["is_correct"]

# Target function
async def run_graph(inputs: dict) -> dict:
    """Run graph and track the trajectory it takes along with the final response."""
    response_text = await run_agent_safely(gemini_agent, inputs['question'])
    return {"response": response_text}


async def main():
    BATCH_START = 42 
    BATCH_END = 48
    
    # 1. Slice your local hardcoded 'examples' list directly in exact code order
    batch_local_examples = examples[BATCH_START:BATCH_END]
    print(f"Running evaluation for local batch items [{BATCH_START} to {BATCH_END}] in exact code order...")

    # 2. Create a temporary, sequential batch dataset name in LangSmith
    batch_dataset_name = f"Batch-{BATCH_START}-to-{BATCH_END}"
    
    if client.has_dataset(dataset_name=batch_dataset_name):
        client.delete_dataset(dataset_name=batch_dataset_name)
        
    batch_dataset = client.create_dataset(dataset_name=batch_dataset_name)
    
    # 3. Upload ONLY this batch's examples sequentially into LangSmith
    for ex in batch_local_examples:
        client.create_example(
            inputs=ex["inputs"],
            outputs=ex["outputs"],
            dataset_id=batch_dataset.id
        )

    # 4. Pass the batch dataset name string into aevaluate
    experiment_results = await client.aevaluate(
        run_graph,
        data=batch_dataset_name,  # Pass the sequential batch dataset name string here
        evaluators=[final_answer_correct],
        experiment_prefix=f"multitool-agent-gemini-evaluation-{BATCH_START}-{BATCH_END}",
        num_repetitions=1,
        max_concurrency=2,
    )
    
    df = experiment_results.to_pandas()
    print("Batch evaluation completed successfully!")
    print(df.head())


if __name__ == "__main__":
    asyncio.run(main())
