package com.health.talan.entities;

import java.io.Serializable;
import java.util.*;

import javax.persistence.CascadeType;
import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.FetchType;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.JoinTable;
import javax.persistence.ManyToMany;
import javax.persistence.OneToMany;
import javax.persistence.Table;
import javax.persistence.Temporal;
import javax.persistence.TemporalType;

import com.fasterxml.jackson.annotation.JsonIgnore;

@Entity
@Table(name = "user")

public class User implements Serializable {

	private static final long serialVersionUID = 1L;

	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Column(name = "id")
	private Long id;

	@Column(name = "firstName")
	private String firstName;

	@Column(name = "lastName")
	private String lastName;

	@Column(name = "mail")
	private String mail;

	@Column(name = "username")
	private String username;

	@Column(name = "password")
	private String password;

	@Temporal(TemporalType.DATE)
	@Column(name = "birthDate")
	private Date birthDate;

	@Column(name = "address")
	private String address;

	@Column(name = "image")
	private PieceJoint image;

	@Column(name = "profession")
	private String profession;

	@Column(name = "professionnalisme")
	private boolean professionnalisme;

	@JsonIgnore
	@ManyToMany(fetch = FetchType.LAZY)
	@JoinTable(name = "user_roles", joinColumns = @JoinColumn(name = "user_id"), inverseJoinColumns = @JoinColumn(name = "role_id"))
	private Set<Role> roles = new HashSet<>();

	@JsonIgnore
	@ManyToMany(cascade = CascadeType.ALL)
	private Set<Interest> interests = new HashSet<>();

	@JsonIgnore
	@OneToMany(cascade = CascadeType.ALL, mappedBy = "user")
	private Set<PieceJustif> pieceJustifs = new HashSet<>();

	@JsonIgnore
	@OneToMany(cascade = CascadeType.ALL, mappedBy = "user")
	private Set<Amis> amis = new HashSet<>();

	@JsonIgnore
	@OneToMany(cascade = CascadeType.ALL, mappedBy = "adminCom", fetch = FetchType.EAGER)
	private Set<Community> myCommunities;
	
	 
	@JsonIgnore
	@ManyToMany(mappedBy="participants" , cascade = CascadeType.ALL)
	private Set<Community> communitiesParticipate;

	@JsonIgnore
	@OneToMany(cascade = CascadeType.ALL, mappedBy = "user")
	private Set<Entreprise> entreprises = new HashSet<>();

	@JsonIgnore
	@OneToMany(mappedBy = "user")
	private Set<Participant> participants = new HashSet<>();

	@JsonIgnore
	@OneToMany(mappedBy = "adminEvent")
	private Set<Event> events = new HashSet<>();

	@JsonIgnore
	@OneToMany(cascade = CascadeType.ALL, mappedBy = "sender")
	private Set<Invitation> invitationsSend = new HashSet<>();

	@JsonIgnore
	@OneToMany(cascade = CascadeType.ALL, mappedBy = "receiver")
	private Set<Invitation> invitationsReceive = new HashSet<>();

	@JsonIgnore
	@OneToMany(cascade = CascadeType.ALL, mappedBy = "user")
	private Set<Publicity> publicities = new HashSet<>();

	@JsonIgnore
	@OneToMany(cascade = CascadeType.ALL, mappedBy = "user")
	private List<Publication> publications = new ArrayList<>();

	@JsonIgnore
	@OneToMany(cascade = CascadeType.ALL, mappedBy = "adminChallenge")
	private Set<Challenge> myChallenge = new HashSet<>();

	@JsonIgnore
	@OneToMany(mappedBy = "user", cascade = CascadeType.ALL)
	private Set<PublicationChallenge> publicationChallenges = new HashSet<>();

	@JsonIgnore
	@OneToMany(mappedBy = "user")
	private Set<Liking> likes = new HashSet<>();

	@JsonIgnore
	@OneToMany(mappedBy = "user")
	private Set<Comment> comments = new HashSet<>();

	public User() {
		super();
	}

	public User(String firstName, String lastName, String mail, String username, String password, Date birthDate,
			String address, String profession, boolean professionnalisme) {

		this.firstName = firstName;
		this.lastName = lastName;
		this.mail = mail;
		this.username = username;
		this.password = password;
		this.birthDate = birthDate;
		this.address = address;
		this.profession = profession;
		this.professionnalisme = professionnalisme;
	}

	public Long getId() {
		return id;
	}

	public void setId(Long id) {
		this.id = id;
	}

	public String getFirstName() {
		return firstName;
	}

	public void setFirstName(String firstName) {
		this.firstName = firstName;
	}

	public String getLastName() {
		return lastName;
	}

	public void setLastName(String lastName) {
		this.lastName = lastName;
	}

	public String getMail() {
		return mail;
	}

	public void setMail(String mail) {
		this.mail = mail;
	}

	public String getUsername() {
		return username;
	}

	public void setUsername(String username) {
		this.username = username;
	}

	public String getPassword() {
		return password;
	}

	public void setPassword(String password) {
		this.password = password;
	}

	public Date getBirthDate() {
		return birthDate;
	}

	public void setBirthDate(Date birthDate) {
		this.birthDate = birthDate;
	}

	public String getAddress() {
		return address;
	}

	public void setAddress(String address) {
		this.address = address;
	}

	public PieceJoint getImage() {
		return image;
	}

	public void setImage(PieceJoint image) {
		this.image = image;
	}

	public String getProfession() {
		return profession;
	}

	public void setProfession(String profession) {
		this.profession = profession;
	}

	public boolean isProfessionnalisme() {
		return professionnalisme;
	}

	public void setProfessionnalisme(boolean professionnalisme) {
		this.professionnalisme = professionnalisme;
	}

	public Set<Role> getRoles() {
		return roles;
	}

	public void setRoles(Set<Role> roles) {
		this.roles = roles;
	}

	public Set<Interest> getInterests() {
		return interests;
	}

	public void setInterests(Set<Interest> interests) {
		this.interests = interests;
	}

	public Set<PieceJustif> getPieceJustifs() {
		return pieceJustifs;
	}

	public void setPieceJustifs(Set<PieceJustif> pieceJustifs) {
		this.pieceJustifs = pieceJustifs;
	}

	public Set<Amis> getAmis() {
		return amis;
	}

	public void setAmis(Set<Amis> amis) {
		this.amis = amis;
	}

	public Set<Community> getMyCommunities() {
		return myCommunities;
	}

	public void setMyCommunities(Set<Community> myCommunities) {
		this.myCommunities = myCommunities;
	}

	public Set<Community> getCommunitiesParticipate() {
		return communitiesParticipate;
	}

	public void setCommunitiesParticipate(Set<Community> communitiesParticipate) {
		this.communitiesParticipate = communitiesParticipate;
	}

	public Set<Entreprise> getEntreprises() {
		return entreprises;
	}

	public void setEntreprises(Set<Entreprise> entreprises) {
		this.entreprises = entreprises;
	}

	public Set<Participant> getParticipants() {
		return participants;
	}

	public void setParticipants(Set<Participant> participants) {
		this.participants = participants;
	}

	public Set<Event> getEvents() {
		return events;
	}

	public void setEvents(Set<Event> events) {
		this.events = events;
	}

	public Set<Invitation> getInvitationsSend() {
		return invitationsSend;
	}

	public void setInvitationsSend(Set<Invitation> invitationsSend) {
		this.invitationsSend = invitationsSend;
	}

	public Set<Invitation> getInvitationsReceive() {
		return invitationsReceive;
	}

	public void setInvitationsReceive(Set<Invitation> invitationsReceive) {
		this.invitationsReceive = invitationsReceive;
	}

	public Set<Publicity> getPublicities() {
		return publicities;
	}

	public void setPublicities(Set<Publicity> publicities) {
		this.publicities = publicities;
	}

	public List<Publication> getPublications() {
		return publications;
	}

	public void setPublications(List<Publication> publications) {
		this.publications = publications;
	}

	public Set<Challenge> getMyChallenge() {
		return myChallenge;
	}

	public void setMyChallenge(Set<Challenge> myChallenge) {
		this.myChallenge = myChallenge;
	}

	public Set<PublicationChallenge> getPublicationChallenges() {
		return publicationChallenges;
	}

	public void setPublicationChallenges(Set<PublicationChallenge> publicationChallenges) {
		this.publicationChallenges = publicationChallenges;
	}

	public Set<Liking> getLikes() {
		return likes;
	}

	public void setLikes(Set<Liking> likes) {
		this.likes = likes;
	}

	public Set<Comment> getComments() {
		return comments;
	}

	public void setComments(Set<Comment> comments) {
		this.comments = comments;
	}

	@Override
	public String toString() {
		return "User [id=" + id + ", firstName=" + firstName + ", lastName=" + lastName + ", mail=" + mail
				+ ", username=" + username + ", password=" + password + ", birthDate=" + birthDate + ", address="
				+ address + ", image=" + image + ", profession=" + profession + ", professionnalisme="
				+ professionnalisme + ", roles=" + roles + ", interests=" + interests + ", pieceJustifs=" + pieceJustifs
				+ ", amis=" + amis + ", myCommunities=" + myCommunities + ", communitiesParticipate="
				+ communitiesParticipate + ", entreprises=" + entreprises + ", participants=" + participants
				+ ", events=" + events + ", invitationsSend=" + invitationsSend + ", invitationsReceive="
				+ invitationsReceive + ", publicities=" + publicities + ", publications=" + publications
				+ ", myChallenge=" + myChallenge + ", publicationChallenges=" + publicationChallenges + ", likes="
				+ likes + ", comments=" + comments + "]";
	}

}
