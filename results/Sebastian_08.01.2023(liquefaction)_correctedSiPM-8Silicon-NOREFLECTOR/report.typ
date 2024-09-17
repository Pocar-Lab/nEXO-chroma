
#import "template.typ": *

#show: project.with(
  title: [Chroma Simulation \ Sebastian\_08.01.2023(liquefaction)\_correctedSiPM \ 8Silicon-NOREFLECTOR],
  authors: ((name: "Pocar Lab", email: "Lab 21", affiliation: "nEXO Collaboration"),),
  date: [#datetime.today().display()],
)

= Experiment Details

- Experiment Name: Sebastian\_08.01.2023(liquefaction)\_correctedSiPM
- Number of Particles: 500000
- Random Seed: 1042
- Excluded Geometry: ['reflector2', 'reflector3', 'reflector4', 'reflector5', 'reflector6', 'reflector7', 'reflector8', 'reflector9']
- PTE: 0.001368 $plus.minus$ 5.2e-05.

#pagebreak()
= Material Properties

== Bulk Materials

#table(
  columns: 5,
  [name], [refractive_index], [absorption_length], [scattering_length], [density],
  [liquid xenon], [1.69], [350000000000.0], [35000000000000], [2.9419999999999997],
  [copper], [0.9733], [0.05], [1000000], [8.96],
  [silicon], [0.682], [1000000.0], [10000000], [2.329],
  [steel], [1.29], [100000000.0], [1000000000], [7.75],
  [teflon], [0.0], [100000000.0], [1000000000], [7.75],
  [silica], [1.6440000000000001], [100000000.0], [1000000000], [2.202],
  [aluminum], [0.09216], [1000000.0], [10000000], [2.7],
  [killing material], [1.0], [100000000.0], [1000000000], [0.0],
)

#table(
  columns: 6,
  [name], [abs(r_i_error)], [eta], [k], [abs(eta_error)], [abs(k_error)],
  [liquid xenon], [0.0], [0.0], [0.0], [0], [0],
  [copper], [0.0], [0.972], [1.5004], [0], [0],
  [silicon], [0.0], [0.8398700000000001], [1.9019], [0], [0],
  [steel], [0.0], [0.0], [0.0], [0], [0],
  [teflon], [0.0], [0.0], [0.0], [0], [0],
  [silica], [0.0], [0.0], [0.0], [0], [0],
  [aluminum], [0.0], [0.09216], [1.9217], [0], [0],
  [killing material], [0.0], [0.0], [0.0], [0], [0],
)

== Surface Properties

#table(
  columns: 6,
  [name], [outer_mat], [inner_mat], [model_id], [reflect_specular], [reflect_diffuse],
  [FBK HD3], [liquid xenon], [silicon], [5], [0], [0],
  [Cu-Xe], [liquid xenon], [copper], [8], [0], [1],
  [silicon-Xe], [liquid xenon], [silicon], [4], [1], [0],
  [killing surface], [None], [None], [8], [0], [0],
)

= Results

== Photon Transmission Efficiency

The Photon Transmission Efficiency (PTE) for this experiment was 
0.001368 $plus.minus$ 5.2e-05.

== Tallies

#table(
  columns: (auto, auto),
  [Tally], [Count],
  
)
#pagebreak()
= Plots

#let plot_files = ("plots/detected_photon_tracks,_seed_1042,_run_1_seed_1042_run_1.png", "plots/reflected_photon_tracks,_seed_1042,_run_1_seed_1042_run_1.png", "plots/incident_vs_emission_angle_seed_1042.png", "plots/emission_angle_distribution_seed_1042_run_1.png", "plots/reflection_angle_distribution_seed_1042.png", "plots/incident_angle_distribution_seed_1042_run_1.png", "plots/detected_photon_tracks,_seed_1042_seed_1042.png", "plots/undetected_photon_tracks,_seed_1042_seed_1042.png", "plots/emission_angle_distribution_seed_1042.png", "plots/reflection_multiplicity_seed_1042.png", "plots/_undetected_photon_tracks,_seed_1042,_run_1_seed_1042_run_1.png", "plots/reflection_multiplicity_seed_1042_run_1.png", "plots/specularly_refelcted_photon_tracks,_seed_1042_seed_1042.png", "plots/filtered_scattered_photon_tracks,_seed_1042,_run_1_seed_1042_run_1.png", "plots/filtered_scattered_photon_tracks,_seed_1042_seed_1042.png", "plots/_diffusively_reflected_photon_tracks,_seed_1042,_run_1_seed_1042_run_1.png", "plots/detected_and_reflected_photon_tracks,_seed_1042,_run_1_seed_1042_run_1.png", "plots/specularly_refelcted_photon_tracks,_seed_1042,_run_1_seed_1042_run_1.png", "plots/position_distribution_seed_1042.png", "plots/incident_vs_emission_angle_seed_1042_run_1.png", "plots/_diffusively_reflected_photon_tracks,_seed_1042_seed_1042.png", "plots/detected_and_reflected_photon_tracks,_seed_1042_seed_1042.png", "plots/photon_tracks,_seed_1042_seed_1042.png", "plots/position_distribution_seed_1042_run_1.png", "plots/incident_angle_distribution_seed_1042.png", "plots/reflection_angle_distribution_seed_1042_run_1.png", "plots/reflected_photon_tracks,_seed_1042_seed_1042.png", "plots/photon_tracks,_seed_1042,_run_1_seed_1042_run_1.png")


#grid(
  columns: 2,
  gutter: 10pt,
  ..plot_files.map(file => {
    figure(
      image(file, width: 100%),
      caption: file.split("/").at(-1).replace("_", " ").replace(".png", "")
    )
  })
)


