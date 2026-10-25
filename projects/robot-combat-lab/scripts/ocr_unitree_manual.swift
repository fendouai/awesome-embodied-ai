import Foundation
import Vision
import ImageIO
let directory=URL(fileURLWithPath:CommandLine.arguments[1])
let files=try FileManager.default.contentsOfDirectory(at:directory,includingPropertiesForKeys:nil).filter{$0.pathExtension=="png"}.sorted{$0.path<$1.path}
for file in files {
 let request=VNRecognizeTextRequest()
 request.recognitionLevel = .accurate
 request.recognitionLanguages=["en-US"]
 let handler=VNImageRequestHandler(url:file,options:[:])
 try handler.perform([request])
 let text=(request.results ?? []).compactMap{$0.topCandidates(1).first?.string}.joined(separator:"\n")
 try text.write(to:file.deletingPathExtension().appendingPathExtension("txt"),atomically:true,encoding:.utf8)
 print(file.lastPathComponent)
}
